package onlyRDD

import java.util

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import org.rogach.scallop.{ScallopConf, ScallopOption}
import weka.core.{Attribute, Instances}
import weka.filters.Filter

import scala.collection.mutable
import scala.reflect.ClassTag


object DistributedFeatureSelectionRDD {


  def main(args: Array[String]): Unit = {

    //Argument parser
    val opts = new ScallopConf(args) {
      banner("\nUsage of this program example: -d connect-4.csv -p 5 CFS,IG,CF  F1,F2 \n\n")
      val dataset: ScallopOption[String] = opt[String]("dataset", required = true, descr = "Dataset to use in CSV format / Class must be last or first column")
      val test_dataset: ScallopOption[String] = opt[String]("test dataset", descr = "Train dataset to use in CSV format / Class must be last or first column")
      val partType: ScallopOption[Boolean] = toggle("vertical", default = Some(false), descrYes = "Vertical partitioning / Default Horizontal")
      val overlap: ScallopOption[Double] = opt[Double]("overlap", default = Some(0.0), descr = "Overlap")
      val class_index: ScallopOption[Boolean] = toggle("first", noshort = true, default = Some(false), descrYes = "Required if class is first column")
      val numParts: ScallopOption[Int] = opt[Int]("partitions", validate = 0 < _, descr = "Num of partitions", required = true)
      val alpha: ScallopOption[Double] = opt[Double]("alpha", descr = "Aplha Value for threshold computation / Default 0.75", validate = { x => 0 <= x && x <= 1 }, default = Some(0.75))
      val fs_algorithms: ScallopOption[String] = opt[String](required = true, default = Some("CFS,IG,RF"), descr = "List of feature selection algorithm")
      val complexity_measure: ScallopOption[String] = opt[String](required = true, default = Some("F1,F2"), descr = "List of complexity measures")
      verify()
    }

    val conf = new SparkConf().setMaster("local[*]").setAppName("Distributed Feature Selection")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val (train_rdd, test_rdd) = createRDDs(opts.dataset(), None, opts.class_index(), sc)
    val attributes = createAttributesMap(train_rdd, sc)
    val br_attributes = sc.broadcast(attributes)

    val fs_algorithms = opts.fs_algorithms().split(",")
    val comp_measures = opts.complexity_measure().split(",")

    val transposed_rdd = if (!opts.partType()) sc.emptyRDD[(Int, Seq[String])] else transposeRDD(train_rdd)
    transposed_rdd.cache()

    var cfs_features_selected = 1


    for (fsa <- fs_algorithms) {

      if (opts.partType()) println(s"*****Using vertical partitioning with ${opts.overlap() * 100}% overlap*****")
      else println(s"*****Using horizontal partitioning*****")


      /** Here we get the votes vector **/
      val (votes, times, cfs_selected) = getVotesVector(train_rdd, br_attributes, opts.numParts(), opts.partType(), opts.overlap(), "CFS", cfs_features_selected, transposed_rdd, sc)


      if (fsa == "CFS") cfs_features_selected = (cfs_selected.sum() / cfs_selected.count()).toInt

      for (compmeasure <- comp_measures) {

        val start_sub_time = System.currentTimeMillis()

        println(s"*****Using $fsa algorithm with $compmeasure as complexity measure*****")
        println(s"*****Number of partitions: ${opts.numParts()}*****")


        val globalCompyMeasure = compmeasure match {
          case "F1" => fisherRatio _
          case "F2" => f2 _
          case _ => zeroGlobal _
        }


        val classifier = compmeasure match {
          case "SVM" =>
            Some(new OneVsRest().setClassifier(new LinearSVC()))
          case "KNN" =>
            Some(new KNNClassifier().setK(1))
          case "DT" =>
            Some(new DecisionTreeClassifier())
          case "NB" =>
            Some(new NaiveBayes())
          case _ =>
            None
        }

        /** Here we get the selected features **/
        val features = computeThreshold(train_rdd, votes, opts.alpha(), classifier, br_attributes, opts.partType(),
          opts.numParts(), 5, transposed_rdd, globalCompyMeasure, sc)
        println(s"Feature selection computation time is ${System.currentTimeMillis() - start_sub_time} (votes + threshold)")


        if (fsa == "CFS") {
          cfs_features_selected = features.count().toInt
        }

        /** Here we evaluate several algorithms with the selected features **/
        val evaluation_time = System.currentTimeMillis()
        evaluateFeatures(train_dataframe, test_dataframe, attributes, inverse_attributes, class_index, features, ss.sparkContext)
        println(s"Evaluation time is ${System.currentTimeMillis() - evaluation_time}")
        println("\n\n")

      }


    }
    println(s"Trainset: ${train_dataframe.count} Testset: ${test_dataframe.count}")
    println(s"Total script time is ${System.currentTimeMillis() - start_time}")

  }


  def createRDDs(train_file: String, test_file: Option[String], class_first_column: Boolean, sc: SparkContext): (RDD[Array[String]], RDD[Array[String]]) = {

    def parse_RDD(rdd: RDD[String], sep: Char): RDD[Array[String]] = rdd.map((x: String) => {
      if (class_first_column) {
        val result = x.split(sep)
        result.drop(1) :+ result(0)
      } else x.split(sep)
    })

    var train_rdd = parse_RDD(sc.textFile(train_file), ',')
    var test_rdd = sc.emptyRDD[Array[String]]

    if (test_file.isDefined) {
      test_rdd = parse_RDD(sc.textFile(test_file.get), ',')
    } else {
      val partitioned = train_rdd.map(row => (row.last, row)).groupByKey()
        .flatMap {
          case (_, value) => value.zipWithIndex
        }
        .map {
          case (row, index) => (index % 3, row)
        }
      train_rdd = partitioned.filter(x => x._1 == 1 || x._1 == 2).map(_._2)
      test_rdd = partitioned.filter(_._1 == 0).map(_._2)


    }
    (train_rdd, test_rdd)
  }

  def createAttributesMap(rdd: RDD[Array[String]], sc: SparkContext): Map[Int, (Option[Set[String]], String)] = {

    /** *****************************
      * Creation of attributes maps
      * *****************************/

    val sample = rdd.takeSample(false, 1)(0).zipWithIndex
    val nominalAttributes = sample.dropRight(1).filter(tuple => parseNumeric(tuple._1).isEmpty).map(_._2) :+ (sample.length - 1)

    val uniques_nominal_values = rdd.flatMap(_.zipWithIndex).filter {
      case (_, index) => nominalAttributes.contains(index)
    }.map(tuple => (tuple._2, tuple._1))
      .combineByKey((value: String) => mutable.Set[String](value),
        (set: mutable.Set[String], new_value: String) => set += new_value,
        (set1: mutable.Set[String], set2: mutable.Set[String]) => set1 ++= set2).map(x => (x._1, Some(x._2.toSet))).collectAsMap()

    sample.map(tuple => {
      if (tuple._2 == (sample.length - 1)) tuple._2 -> (uniques_nominal_values.getOrElse(tuple._2, None), "class")
      else tuple._2 -> (uniques_nominal_values.getOrElse(tuple._2, None), "att_" + tuple._2.toString)
    }).toMap
  }

  def getVotesVector(rdd: RDD[Array[String]], br_attributes: Broadcast[Map[Int, (Option[Set[String]], String)]],
                     numParts: Int, vertical: Boolean, overlap: Double, filter: String, ranking_features: Int,
                     transpose_input: RDD[(Int, Seq[String])],
                     sc: SparkContext): (RDD[(String, Int)], RDD[Long], RDD[Int]) = {

    /** **************************
      * Getting the Votes vector.
      * **************************/

    val rounds = 5
    var times = sc.emptyRDD[Long]
    var cfs_selected = sc.emptyRDD[Int]


    val votes = {
      var sub_votes = sc.emptyRDD[(String, Int)]
      if (vertical) {
        // Get the class column
        val br_class_column = sc.broadcast(transpose_input.filter {
          case (columnindex, row) => columnindex == br_attributes.value.size - 1
        }.first())
        val rdd_no_class_column = transpose_input.filter {
          case (columnindex, row) => columnindex != br_attributes.value.size - 1
        }
        val br_classes = sc.broadcast(br_attributes.value(br_attributes.value.size - 1)._1.get)
        for (_ <- 1 to rounds) {
          val result = verticalPartitioningFeatureSelection(sc, shuffleRDD(rdd_no_class_column),
            br_class_column, br_classes, br_attributes, numParts, filter, overlap, ranking_features)
          sub_votes = sub_votes ++ result.map(x => (x._1, x._2._1))
          times = times ++ result.map(x => x._2._2)
          if (filter == "CFS") {
            cfs_selected = cfs_selected ++ result.map(x => x._2._3)
          }
        }
      } else {
        val (schema, class_schema_index) = WekaWrapperRDD.attributesSchema(br_attributes.value)
        val br_schema = sc.broadcast(schema)
        for (_ <- 1 to rounds) {
          val result = horizontalPartitioningFeatureSelection(sc, shuffleRDD(rdd),
            br_attributes, numParts, filter, br_schema, class_schema_index, ranking_features)
          sub_votes = sub_votes ++ result.map(x => (x._1, x._2._1))
          times = times ++ result.map(x => x._2._2)
          if (filter == "CFS") {
            cfs_selected = cfs_selected ++ result.map(x => x._2._3)
          }
        }
      }
      sub_votes.reduceByKey(_ + _)
    }
    (votes, times, cfs_selected)
  }

  def computeThreshold(rdd: RDD[Array[String]], votes: RDD[(String, Int)], alpha_value: Double, classifier: Option[PipelineStage],
                       br_attributes: Broadcast[Map[Int, (Option[Set[String]], String)]],
                       vertical: Boolean, numParts: Int, rounds: Int = 5, transpose_input: RDD[(Int, Seq[String])],
                       globalComplexityMeasure: (RDD[Array[String]], Broadcast[Map[Int, (Option[Set[String]], String)]], SparkContext, RDD[(Int, Seq[String])]) => Double,
                       sc: SparkContext): RDD[String] = {

    /** ******************************************
      * Computing 'Selection Features Threshold'
      * ******************************************/

    val votes_length = votes.count()
    val threshold_time = System.currentTimeMillis()
    val avg_votes = votes.map(_._2).sum / votes_length
    val std_votes = math.sqrt(votes.map(votes => math.pow(votes._2 - avg_votes, 2)).sum / votes_length)
    val minVote = if (vertical) rounds * (numParts - 1) else (avg_votes - (std_votes / 2)).toInt
    val maxVote = if (vertical) rounds * numParts else (avg_votes + (std_votes / 2)).toInt

    //We get the features that aren't in the votes set. That means features -> Votes = 0
    // ****Class column included****
    val selected_features_0_votes = sc.parallelize(br_attributes.value.map(_._2._2).filter( !votes.map(_._1).collect().contains(_)).toArray)


    val alpha = alpha_value
    var e_v = collection.mutable.ArrayBuffer[(Int, Double)]()


    var compMeasure = 0.0
    val step = if (vertical) 1 else 5
    for (a <- minVote to maxVote by step) {
      val starting_time = System.currentTimeMillis()
      // We add votes below Threshold value
      val selected_features = selected_features_0_votes ++ votes.filter(_._2 < a).map(_._1)

      if (selected_features.count() > 1) {
        //        println(s"\nStarting threshold computation with minVotes = $a / maxVotes = $maxVote with ${selected_features.count() - 1} features")
        val selected_features_rdd = dataframe.select(selected_features.collect().map(col): _*)
        val retained_feat_percent = (selected_features.count().toDouble / dataframe.columns.length) * 100
        if (classifier.isDefined) {
          val (pipeline_stages, columns_to_cast) = createPipeline(sc.parallelize(selected_features_dataframe.columns), br_attributes, sc)
          val casted_dataframe = castDFToDouble(selected_features_dataframe, columns_to_cast)
          val pipeline = new Pipeline().setStages(pipeline_stages :+ classifier.get).fit(casted_dataframe)
          val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
            .setPredictionCol("prediction").setMetricName("accuracy")
          compMeasure = 1 - evaluator.evaluate(pipeline.transform(casted_dataframe))
        } else {
          val start_comp = System.currentTimeMillis()
          compMeasure = globalComplexityMeasure(selected_features_dataframe, br_attributes, sc, transpose_input, class_index)
          //          println(s"\n\tComplexity Measure Computation Time: ${System.currentTimeMillis() - start_comp}.")
        }

        e_v += ((a, alpha * compMeasure + (1 - alpha) * retained_feat_percent))

        //        println(s"\tThreshold computation in ${System.currentTimeMillis() - starting_time} " +
        //          s"\n\t\t Complexity Measure Value: $compMeasure \n\t\t Retained Features Percent: $retained_feat_percent " +
        //          s"\n\t\t EV Value = ${alpha * compMeasure + (1 - alpha) * retained_feat_percent} \n")

      }
    }

    val selected_threshold = e_v.minBy(_._2)._1
    val features = selected_features_0_votes ++ votes.filter(_._2 < selected_threshold).map(_._1)


    println(s"\nTotal threshold computation in ${System.currentTimeMillis() - threshold_time}")
    println(s"Number of features is ${features.count() - 1}")

    features
  }

  /** ***********************
    * Partitioning functions
    * ************************/

  def horizontalPartitioningFeatureSelection(sc: SparkContext, input: RDD[Array[String]],
                                             br_attributes: Broadcast[Map[Int, (Option[Set[String]], String)]],
                                             numParts: Int, filter: String,
                                             br_attributes_schema: Broadcast[util.ArrayList[Attribute]],
                                             class_schema_index: Int, ranking_features: Int): RDD[(String, (Int, Long, Int))] = {

    /** Horizontally partition selection features */

    input.map(row => (row.last, row)).groupByKey()
      .flatMap({
        // Add an index for each subset (keys)
        case (_, value) => value.zipWithIndex
      })
      .map {
        // Get the partition number for each row and make it the new key
        case (row: Array[String], index: Int) => (index % numParts, row)
      }
      .combineByKey(
        (row: Array[String]) => {
          val data = new Instances("Rel", br_attributes_schema.value, 0)
          data.setClassIndex(class_schema_index)
          WekaWrapperRDD.addRowToInstances(data, br_attributes.value, br_attributes_schema.value, row)
        },
        (inst: Instances, row: Array[String]) => WekaWrapperRDD.addRowToInstances(inst, br_attributes.value, br_attributes_schema.value, row),
        (inst1: Instances, inst2: Instances) => WekaWrapperRDD.mergeInstances(inst1, inst2)
      )
      .flatMap {
        case (_, inst) =>
          val start_time = System.currentTimeMillis()
          val filtered_data = Filter.useFilter(inst, WekaWrapperRDD.filterAttributes(inst, filter, ranking_features))
          val time = System.currentTimeMillis() - start_time
          val selected_attributes = WekaWrapperRDD.getAttributes(filtered_data)
          (br_attributes.value.values.map(_._2).toSet.diff(selected_attributes) - br_attributes.value(br_attributes.value.size - 1)._2).map((_, (1, time, filtered_data.numAttributes())))

      }.reduceByKey((t1, t2) => (t1._1 + t2._1, math.max(t1._2, t2._2), math.max(t1._3, t2._3)))


  }


  def verticalPartitioningFeatureSelection(sc: SparkContext, transposed: RDD[(Int, Seq[String])], br_class_column: Broadcast[(Int, Seq[String])], br_classes: Broadcast[Set[String]],
                                           br_attributes: Broadcast[Map[Int, (Option[Set[String]], String)]],
                                           numParts: Int, filter: String, overlap: Double = 0, ranking_features: Int): RDD[(String, (Int, Long, Int))] = {

    var overlapping: RDD[(Int, Seq[String])] = sc.emptyRDD[(Int, Seq[String])]
    if (overlap > 0) {
      overlapping = transposed.sample(withReplacement = false, overlap)
    }
    val br_overlapping = sc.broadcast(overlapping.collect().toIterable)

    //Remove the class column and assign a partition to each column
    transposed.subtract(overlapping)
      .zipWithIndex.map(line => (line._2 % numParts, line._1))
      .groupByKey().flatMap {
      case (_, iter) =>
        val start_time = System.currentTimeMillis()
        val data = WekaWrapperRDD.createInstancesFromTranspose(iter ++ br_overlapping.value, br_attributes.value, br_class_column.value, br_classes.value)
        val filtered_data = Filter.useFilter(data, WekaWrapperRDD.filterAttributes(data, filter, ranking_features))
        val time = System.currentTimeMillis() - start_time
        val selected_attributes = WekaWrapperRDD.getAttributes(filtered_data)

        // Getting the diff we can obtain the features to increase the votes and taking away the class
        (br_attributes.value.values.map(_._2).toSet.diff(selected_attributes) - br_attributes.value(br_attributes.value.size - 1)._2).map((_, (1, time, filtered_data.numAttributes())))


    }.reduceByKey((t1, t2) => (t1._1 + t2._1, math.max(t1._2, t2._2), math.max(t1._3, t2._3)))


  }

  /** ********************
    * Complexity measures
    * *********************/


  def zeroGlobal(rdd: RDD[Array[String]], br_attributes: Broadcast[Map[Int, (Option[Set[String]], String)]], sc: SparkContext, transposedRDD: RDD[(Int, Seq[String])]): Double = {
    0.0
  }

  def fisherRatio(rdd: RDD[Array[String]], br_attributes: Broadcast[Map[Int, (Option[Set[String]], String)]], sc: SparkContext, transposedRDD: RDD[(Int, Seq[String])]): Double = {


    // ProportionclassMap => Class -> Proportion of class
    val class_name = br_attributes.value(class_index)._2
    val samples = dataframe.count().toDouble
    val br_proportionClassMap = sc.broadcast(dataframe.groupBy(class_name).count().rdd.map(row => row(0) -> (row(1).asInstanceOf[Long] / samples.toDouble)).collect().toMap)
    val class_column = sc.broadcast(dataframe.select(class_name).rdd.map(_ (0)).collect())
    val br_columns = sc.broadcast(dataframe.columns)

    val rdd = if (!transposedRDD.isEmpty) transposedRDD.filter { case (x, _) => br_columns.value.contains(br_attributes.value(x)._2) } else transposeRDD(dataframe.drop(class_name).rdd)
    val f1 = rdd.map {
      case (column_index, row) =>
        val zipped_row = row.zip(class_column.value)
        var sumMean: Double = 0
        var sumVar: Double = 0

        //Auxiliar Arrays
        val computed_classes = collection.mutable.ArrayBuffer.empty[String]

        if (br_attributes.value(column_index)._1.isDefined) {
          //If we have categorical values we need to discretize them.
          // We use zipWithIndex where its index is its discretize value.
          val values = br_attributes.value(column_index)._1.get.zipWithIndex.map { case (value, sub_index) => value -> (sub_index + 1) }.toMap
          br_proportionClassMap.value.keySet.foreach { _class_ =>
            val filtered_class_column = zipped_row.filter(_._2 == _class_)
            val mean_class = filtered_class_column.groupBy(_._1).map { case (key, value) => value.length * values(key.toString) }.sum / samples.toDouble

            computed_classes += _class_.toString

            br_proportionClassMap.value.keySet.foreach { sub_class_ =>
              if (!computed_classes.contains(sub_class_)) {
                val mean_sub_class = zipped_row.filter(_._2 == sub_class_).groupBy(_._1).map { case (key, value) => value.length * values(key.toString) }.sum / samples.toDouble

                sumMean += scala.math.pow(mean_class - mean_sub_class, 2) * br_proportionClassMap.value(_class_) * br_proportionClassMap.value(sub_class_)
              }
            }
            val variance = filtered_class_column.map(value => math.pow(values(value._1.toString) - mean_class, 2)).sum / samples.toDouble
            sumVar += variance * br_proportionClassMap.value(_class_)
          }
        } else {

          br_proportionClassMap.value.keySet.foreach { _class_ =>
            val filtered_class_column = zipped_row.filter(_._2 == _class_)
            val mean_class = filtered_class_column.map(_._1.toString.toDouble).sum / samples.toDouble

            computed_classes += _class_.toString

            br_proportionClassMap.value.keySet.foreach { sub_class_ =>
              if (!computed_classes.contains(sub_class_)) {
                val mean_sub_class = zipped_row.filter(_._2 == sub_class_).map(_._1.toString.toDouble).sum / samples.toDouble

                sumMean += scala.math.pow(mean_class - mean_sub_class, 2) * br_proportionClassMap.value(_class_) * br_proportionClassMap.value(sub_class_)
              }
            }
            val variance = filtered_class_column.map(value => math.pow(value._1.toString.toDouble - mean_class, 2)).sum / samples.toDouble
            sumVar += variance * br_proportionClassMap.value(_class_)


          }
        }
        sumMean / sumVar
    }.max
    1 / f1
  }

  def f2(rdd: RDD[Array[String]], br_attributes: Broadcast[Map[Int, (Option[Set[String]], String)]], sc: SparkContext, transposedRDD: RDD[(Int, Seq[String])]): Double = {
    val class_name = br_attributes.value(class_index)._2
    val class_column = sc.broadcast(dataframe.select(class_name).rdd.map(_ (0)).collect())
    val br_columns = sc.broadcast(dataframe.columns)
    val rdd = if (!transposedRDD.isEmpty) transposedRDD.filter { case (x, _) => br_columns.value.contains(br_attributes.value(x)._2) } else transposeRDD(dataframe.drop(class_name).rdd)


    val f2 = rdd.map {

      case (column_index, row) =>
        var result = 0.0
        val zipped_row = row.zip(class_column.value)

        //Auxiliar Arrays
        val computed_classes = collection.mutable.ArrayBuffer.empty[String]

        if (br_attributes.value(column_index)._1.isDefined) {
          //If we have categorical values we need to discretize them.
          // We use zipWithIndex where its index is its discretize value.
          val values = br_attributes.value(column_index)._1.get.zipWithIndex.map { case (value, sub_index) => value -> (sub_index + 1) }.toMap


          var minmaxi = 0
          var maxmini = 0
          var maxmaxi = 0
          var minmini = 0

          br_attributes.value(class_index)._1.get.foreach { _class_ =>

            val datasetC = zipped_row.filter(_._2 == _class_).map(x => values(x._1.toString))
            computed_classes += _class_.toString

            br_attributes.value(class_index)._1.get.foreach { sub_class_ =>
              if (!computed_classes.contains(sub_class_)) {
                val datasetK = zipped_row.filter(_._2 == sub_class_).map(x => values(x._1.toString))

                minmaxi = Seq(datasetC.max, datasetK.max).min
                maxmini = Seq(datasetC.min, datasetK.min).max
                maxmaxi = Seq(datasetC.max, datasetK.max).max
                minmini = Seq(datasetC.min, datasetK.min).min

              }


            }
            val div = if (maxmaxi - minmini == 0) 0.01 else maxmaxi - minmini
            result += Seq(0, minmaxi - maxmini).max / div

          }
        } else {
          br_attributes.value(class_index)._1.get.foreach { _class_ =>
            val datasetC = zipped_row.filter(_._2 == _class_).map(_._1.toString.toDouble)

            var minmaxi = 0.0
            var maxmini = 0.0
            var maxmaxi = 0.0
            var minmini = 0.0

            br_attributes.value(class_index)._1.get.foreach { sub_class_ =>
              if (!computed_classes.contains(sub_class_)) {
                val datasetK = zipped_row.filter(_._2 == sub_class_).map(_._1.toString.toDouble)

                minmaxi = Seq(datasetC.max, datasetK.max).min
                maxmini = Seq(datasetC.min, datasetK.min).max
                maxmaxi = Seq(datasetC.max, datasetK.max).max
                minmini = Seq(datasetC.min, datasetK.min).min
              }

            }
            val div = if (maxmaxi - minmini == 0) 0.01 else maxmaxi - minmini
            result += Seq(0, minmaxi - maxmini).max / div
          }
        }
        result
    }
    f2.sum()

  }


  /** *******************
    * Auxiliar Functions
    * ********************/

  def parseNumeric(s: String): Option[Double] = {
    try {
      Some(s.toDouble)
    } catch {
      case _: Exception => None
    }
  }

  def transposeRDD(rdd: RDD[Array[String]]): RDD[(Int, Seq[String])] = {
    val columnAndRow = rdd.zipWithIndex.flatMap {
      case (row, rowIndex) => row.toSeq.zipWithIndex.map {
        case (element, columnIndex) => columnIndex -> (rowIndex, element)
      }
    }
    columnAndRow.groupByKey.sortByKey().map {
      case (columnIndex, rowIterable) => (columnIndex, rowIterable.toSeq.sortBy(_._1).map(_._2))
    }
  }

  def shuffleRDD[B: ClassTag](rdd: RDD[B]): RDD[B] = {
    rdd.mapPartitions(iter => {
      val rng = new scala.util.Random()
      iter.map((rng.nextInt, _))
    }).partitionBy(new HashPartitioner(rdd.partitions.length)).values
  }

}