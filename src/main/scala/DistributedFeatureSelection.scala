import org.apache.spark.{HashPartitioner, SparkContext}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.col
import org.rogach.scallop.{ScallopConf, ScallopOption, Subcommand}
import weka.attributeSelection.{CfsSubsetEval, GreedyStepwise, InfoGainAttributeEval, Ranker}
import weka.filters.Filter
import weka.filters.supervised.attribute.AttributeSelection

import scala.reflect.ClassTag


object DistributedFeatureSelection {


  def main(args: Array[String]): Unit = {
    //Argument parser
    //TODO: Parse weka feature selection algorithm
    val opts = new ScallopConf(args) {
      banner("\nUsage of this program: -d file -p number_of_partitions -v true_if_vertical_partition(default = false) measure (classifier -m classifier | -o another classifier) \n\n" +
        "Examples:  -d connect-4.data -p 10 measure classifier -m SVM \n\t\t   -d connect-4.data -p 10 measure -o F1 \n"
      )
      val dataset: ScallopOption[String] = opt[String]("dataset", required = true, descr = "Dataset to use in CSV format / Class must be last column")
      val partType: ScallopOption[Boolean] = toggle("vertical", default = Some(false), descrYes = "Vertical partitioning / Default Horizontal")
      val numParts: ScallopOption[Int] = opt[Int]("partitions", validate = 0 <, descr = "Num of partitions", required = true)
      val compMeasure = new Subcommand("measure") {
        val classifier = new Subcommand("classifier") {
          val model: ScallopOption[String] = opt[String]("model", descr = "Available Classifiers:  SVM, Knn, Decision Tree (DT), NaiveBayes (NB)")
        }
        addSubcommand(classifier)
        val other: ScallopOption[String] = opt[String]("other", descr = "Available Metrics: F1")

      }
      addSubcommand(compMeasure)
      val alpha: ScallopOption[Double] = opt[Double]("alpha", descr = "Aplha Value for threshold computation / Default 0.75", validate = x => 0 <= x && x <= 1, default = Some(0.75))
      verify()
    }

    val globalCompyMeasure = opts.compMeasure.other.getOrElse("None") match {
      case "F1" => fisherRatio _
      case _ => zeroGlobal _
    }


    val classifier = opts.compMeasure.classifier.model.getOrElse("None") match {
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

    val selected_features = select_features(opts.dataset(), opts.numParts(), opts.partType(), opts.alpha(), globalCompyMeasure, classifier)

    println(selected_features)


  }

  def select_features(dataset_file: String, numParts: Int, vertical: Boolean = false, alpha_value: Double,
                      globalComplexityMeasure: (DataFrame, Map[Int, (Option[Seq[String]], String)], SparkContext, Option[RDD[(Int, Seq[Any])]]) => Double,
                      classifier: Option[PipelineStage]): Set[String] = {

    val ss = SparkSession.builder().appName("distributed_feature_selection").master("local[*]").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")
    val dataframe = ss.read.option("maxColumns", "30000").csv(dataset_file)


    /** *****************************
      * Creation of attributes maps
      * *****************************/

    val class_index = dataframe.columns.length - 1
    val first_row = dataframe.first().toSeq.map(_.toString)
    val inverse_attributes_ = collection.mutable.Map[String, Int]()

    val attributes = dataframe.columns.zipWithIndex.map({
      case (column_name, index) =>
        // If categorical we need to add the distinct values it can take plus its column name
        if (parseNumeric(first_row(index)).isEmpty || index == class_index) {
          inverse_attributes_ += column_name -> index
          index -> (Some(dataframe.select(dataframe.columns(index)).distinct().collect().toSeq.map(_.get(0).toString)), column_name)
        } else {
          // If not categorical we only need column name
          inverse_attributes_ += column_name -> index
          index -> (None, column_name)
        }
    }).toMap


    val inverse_attributes = inverse_attributes_.toMap

    /** **************************
      * Getting the Votes vector.
      * **************************/

    val rounds = 5
    val start_time = System.currentTimeMillis()
    if (vertical) println("*****Using vertical partitioning*****") else println("*****Using horizontal partitioning*****")

    var transpose_input: Option[RDD[(Int, Seq[Any])]] = None

    val votes = {
      var sub_votes = Array[(String, Int)]()
      if (vertical) {
        transpose_input = Some(transposeRDD(dataframe.rdd))
        for (round <- 1 to rounds) {
          println(s"Round: $round")
          sub_votes = sub_votes ++ vertical_partitioning_feature_selection(ss.sparkContext, shuffleRDD(transpose_input.get), attributes, inverse_attributes, class_index, numParts)
        }
      } else {
        for (round <- 1 to rounds) {
          println(s"Round: $round")
          sub_votes = sub_votes ++ horizontal_partitioning_feature_selection(ss.sparkContext, shuffleRDD(dataframe.rdd), attributes, inverse_attributes, class_index, numParts)
        }
      }
      sub_votes.groupBy(_._1).map(tuple => (tuple._1, tuple._2.map(_._2).sum)).toSeq
    }

    println(s"Votes computation time:${System.currentTimeMillis() - start_time}")

    /** ******************************************
      * Computing 'Selection Features Threshold'
      * ******************************************/

    val avg_votes = votes.map(_._2).sum.toDouble / votes.length
    val std_votes = math.sqrt(votes.map(votes => math.pow(votes._2 - avg_votes, 2)).sum / votes.length)
    val minVote = if (vertical) rounds * (numParts - 1) else (avg_votes - (std_votes / 2)).toInt
    val maxVote = if (vertical) rounds * numParts else (avg_votes + (std_votes / 2)).toInt

    //We get the features that aren't in the votes set. That means features -> Votes = 0
    // ****Class column included****
    val selected_features_0_votes = inverse_attributes.keySet.diff(votes.map(_._1).toSet)

    val alpha = alpha_value
    var e_v = collection.mutable.ArrayBuffer[(Int, Double)]()

    val start_fisher = System.currentTimeMillis()
    var compMeasure = globalComplexityMeasure(dataframe, attributes, ss.sparkContext, transpose_input)
    println(s"Complexity Measure Computation Time: ${System.currentTimeMillis() - start_fisher}. Value $compMeasure")

    val step = if (vertical) 1 else 5
    for (a <- minVote to maxVote by step) {
      val starting_time = System.currentTimeMillis()
      // We add votes below Threshold value
      val selected_features = (selected_features_0_votes ++ votes.filter(_._2 < a).map(_._1)).toSeq

      if (selected_features.length > 1) {
        println(s"Starting threshold computation with minVotes = $a / maxVotes = $maxVote with ${selected_features.length - 1} features")
        val selected_features_dataframe = dataframe.select(selected_features.head, selected_features.tail: _*)
        val retained_feat_percent = (selected_features.length.toDouble / dataframe.columns.length) * 100
        if (classifier.isDefined)
          compMeasure = classification_error(selected_features_dataframe, attributes, inverse_attributes, class_index, classifier.get)
        e_v += ((a, alpha * compMeasure + (1 - alpha) * retained_feat_percent))

        println(s"\tThreshold computation in ${System.currentTimeMillis() - starting_time} " +
          s"\n\t\t Error: $compMeasure \n\t\t Retained Features Percent: $retained_feat_percent " +
          s"\n\t\t EV Value = ${alpha * compMeasure + (1 - alpha) * retained_feat_percent} \n")

      }
    }

    val selected_threshold = e_v.minBy(_._2)._1
    (selected_features_0_votes ++ votes.filter(_._2 < selected_threshold).map(_._1)) - attributes(class_index)._2


    //      #For use with Weka library
    //      val selected_inverse_features_map = inverse_attributes.filterKeys(selected_features.contains(_))
    //      val selected_features_map = attributes.filterKeys(selected_inverse_features_map.values.toSeq.contains(_))
    //      WekaWrapper.createInstances(df, selected_features_map, selected_inverse_features_map, class_index)

  }


  /** ***********************
    * Partitioning functions
    * ************************/

  def horizontal_partitioning_feature_selection(sc: SparkContext, input: RDD[Row],
                                                attributes: Map[Int, (Option[Seq[String]], String)], inverse_attributes: Map[String, Int],
                                                class_index: Int, numParts: Int): Array[(String, Int)] = {

    /** Horizontally partition selection features */


    val br_attributes = sc.broadcast(attributes)
    val br_inverse_attributes = sc.broadcast(inverse_attributes)

    val partitioned = input.map(row => (row.get(row.length - 1), row)).groupByKey()
      .flatMap({
        // Add an index for each subset (keys)
        case (_, value) => value.zipWithIndex //scala.util.Random.shuffle(value).zipWithIndex
      })
      .map({
        // Get the partition number for each row and make it the new key
        case (row, index) => (index % numParts, row)
      })

    partitioned.groupByKey().flatMap { case (_, iter) =>

      val data = WekaWrapper.createInstances(iter, br_attributes.value, class_index)

      //Run Weka Filter to FS
      val filter = new AttributeSelection

      val eval = new CfsSubsetEval
      val search = new GreedyStepwise
      search.setSearchBackwards(true)
      filter.setEvaluator(eval)
      filter.setSearch(search)
      filter.setInputFormat(data)
      val filtered_data = Filter.useFilter(data, filter)
      val selected_attributes = WekaWrapper.getAttributes(filtered_data)


      //      val filter2 = new AttributeSelection
      //      val eval2 = new InfoGainAttributeEval
      //      val search2 = new Ranker()
      //      search2.setNumToSelect(selected_attributes.size)
      //      filter2.setEvaluator(eval2)
      //      filter2.setSearch(search2)
      //      filter2.setInputFormat(data)
      //      val filtered_data2 = Filter.useFilter(data, filter2)

      // Getting the diff we can obtain the features to increase the votes and taking away the class
      (br_inverse_attributes.value.keySet.diff(selected_attributes) - br_attributes.value(class_index)._2).map((_, 1))

    }.reduceByKey(_ + _).collect()

  }

  def vertical_partitioning_feature_selection(sc: SparkContext, transposed: RDD[(Int, Seq[Any])],
                                              attributes: Map[Int, (Option[Seq[String]], String)], inverse_attributes: Map[String, Int],
                                              class_index: Int, numParts: Int): Array[(String, Int)] = {

    //TODO: Overlap

    /** Vertically partition selection features */

    val br_attributes = sc.broadcast(attributes)
    val br_inverse_attributes = sc.broadcast(inverse_attributes)
    val classes = attributes(class_index)._1.get
    val br_classes = sc.broadcast(classes)

    // Get the class column
    val br_class_column = sc.broadcast(transposed.filter { case (columnindex, _) => columnindex == class_index }.first())

    //Remove the class column and assign a partition to each column
    transposed.filter { case (columnindex, _) => columnindex != class_index }
      .zipWithIndex.map(line => (line._2 % numParts, line._1))
      .groupByKey().flatMap {
      case (_, iter) =>

        val data = WekaWrapper.createInstancesFromTranspose(iter, br_attributes.value, br_class_column.value, br_classes.value)

        //Run Weka Filter to FS
        val filter = new AttributeSelection

        val eval = new CfsSubsetEval
        val search = new GreedyStepwise
        search.setSearchBackwards(true)

        filter.setEvaluator(eval)
        filter.setSearch(search)
        filter.setInputFormat(data)
        val filtered_data = Filter.useFilter(data, filter)
        val selected_attributes = WekaWrapper.getAttributes(filtered_data)


        //        val filter2 = new AttributeSelection
        //        val eval2 = new InfoGainAttributeEval
        //        val search2 = new Ranker()
        //        search2.setNumToSelect(selected_attributes.size)
        //        filter2.setEvaluator(eval2)
        //        filter2.setSearch(search2)
        //        filter2.setInputFormat(data)
        //        val filtered_data2 = Filter.useFilter(data, filter2)


        // Getting the diff we can obtain the features to increase the votes and taking away the class
        (br_inverse_attributes.value.keySet.diff(selected_attributes) - br_attributes.value(class_index)._2).map((_, 1))


    }.reduceByKey(_ + _).collect()


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

  def transposeRDD(rdd: RDD[Row]): RDD[(Int, Seq[Any])] = {
    val columnAndRow = rdd.zipWithIndex.flatMap {
      case (row, rowIndex) => row.toSeq.zipWithIndex.map {
        case (element, columnIndex) => columnIndex -> (rowIndex, element)
      }
    }
    columnAndRow.groupByKey.sortByKey().map { case (columnIndex, rowIterable) => (columnIndex, rowIterable.toSeq.sortBy(_._1).map(_._2)) }

  }

  def shuffleRDD[B: ClassTag](rdd: RDD[B]): RDD[B] = {

    rdd.mapPartitions(iter => {
      val rng = new scala.util.Random()
      iter.map((rng.nextInt, _))
    }).partitionBy(new HashPartitioner(rdd.partitions.length)).values

  }


  /** ********************
    * Complexity measures
    * *********************/

  def classification_error(selected_features_dataframe: DataFrame,
                           attributes: Map[Int, (Option[Seq[String]], String)],
                           inverse_attributes: Map[String, Int],
                           class_index: Int,
                           classifier: PipelineStage): Double = {

    //Creation of pipeline
    var pipeline: Array[PipelineStage] = Array()

    //Mllib needs to special columns [features] and [label] to work. We have to assemble the columns we selected
    var columns_to_assemble: Array[String] = selected_features_dataframe.columns.filter {
      cname =>
        val original_attr_index = inverse_attributes(cname)
        attributes(original_attr_index)._1.isEmpty && original_attr_index != class_index
    }

    //Cast to double of columns that aren't categorical
    val df = selected_features_dataframe.select(selected_features_dataframe.columns.map { c =>
      if (columns_to_assemble.contains(c)) {
        col(c).cast("Double")
      } else {
        col(c)
      }
    }: _*)


    // Transform categorical data to one_hot in order to work with MLlib
    df.columns.filter {
      cname =>
        val original_attr_index = inverse_attributes(cname)
        attributes(original_attr_index)._1.isDefined && original_attr_index != class_index
    }.foreach {
      cname =>
        val st_indexer = new StringIndexer().setInputCol(cname).setOutputCol(s"${cname}_index")
        pipeline = pipeline :+ st_indexer
        pipeline = pipeline :+ new OneHotEncoder().setInputCol(st_indexer.getOutputCol).setOutputCol(s"${cname}_vect")
        columns_to_assemble = columns_to_assemble :+ s"${cname}_vect"
    }

    //Transform class column from categorical to index
    pipeline = pipeline :+ new StringIndexer().setInputCol(attributes(class_index)._2).setOutputCol("label")
    //Assemble features
    pipeline = pipeline :+ new VectorAssembler().setInputCols(columns_to_assemble).setOutputCol("features")

    val df_one_hot = new Pipeline().setStages(pipeline :+ classifier).fit(df).transform(df)
    

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    1 - evaluator.evaluate(df_one_hot)


  }

  def zeroGlobal(dataframe: DataFrame, attributes: Map[Int, (Option[Seq[String]], String)], sc: SparkContext, transposedRDD: Option[RDD[(Int, Seq[Any])]]): Double = {
    0.0
  }

  def fisherRatio(dataframe: DataFrame, attributes: Map[Int, (Option[Seq[String]], String)], sc: SparkContext, transposedRDD: Option[RDD[(Int, Seq[Any])]]): Double = {


    // ProportionclassMap => Class -> Proportion of class
    val samples = dataframe.count().toDouble
    val br_proportionClassMap = sc.broadcast(dataframe.groupBy(dataframe.columns.last).count().rdd.map(row => row(0) -> (row(1).asInstanceOf[Long] / samples.toDouble)).collect().toMap)
    val br_attributes = sc.broadcast(attributes)
    val class_column = dataframe.select(dataframe.columns.last).rdd.map(_ (0)).collect()


    val rdd = if (transposedRDD.isDefined) transposedRDD.get else transposeRDD(dataframe.drop(dataframe.columns.last).rdd)
    val f1 = rdd.map {
      case (column_index, row) =>
        val zipped_row = row.zip(class_column)
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


}



