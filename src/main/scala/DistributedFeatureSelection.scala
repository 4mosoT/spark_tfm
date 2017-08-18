import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, collect_set}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.{HashPartitioner, SparkContext}
import org.rogach.scallop.{ScallopConf, ScallopOption, Subcommand}
import weka.filters.Filter

import scala.collection.mutable
import scala.reflect.ClassTag


object DistributedFeatureSelection {


  def main(args: Array[String]): Unit = {

    //Argument parser
    val opts = new ScallopConf(args) {
      banner("\nUsage of this program: -d file -p number_of_partitions -v true_if_vertical_partition(default = false) measure (classifier -m classifier | -o another classifier) \n\n" +
        "Examples:  -d connect-4.data -p 10 measure classifier -m SVM \n\t\t   -d connect-4.data -p 10 measure -o F1 \n"
      )
      val dataset: ScallopOption[String] = opt[String]("dataset", required = true, descr = "Dataset to use in CSV format / Class must be last or first column")
      val test_dataset: ScallopOption[String] = opt[String]("test dataset", descr = "Train dataset to use in CSV format / Class must be last or first column")

      val class_index: ScallopOption[Boolean] = toggle("first", noshort = true, default = Some(false), descrYes = "Required if class is first column")
      val feature_algorithm: ScallopOption[String] = opt[String]("feature_selection_algorithm", required = true, descr = "Feature selection algorithm",
        validate = { x => x == "CFS" || x == "IG" || x == "RF" })
      val partType: ScallopOption[Boolean] = toggle("vertical", default = Some(false), descrYes = "Vertical partitioning / Default Horizontal")
      val overlap: ScallopOption[Double] = opt[Double]("overlap", default = Some(0.0), descr = "Overlap")
      val numParts: ScallopOption[Int] = opt[Int]("partitions", validate = 0 < _, descr = "Num of partitions", required = true)
      val compMeasure = new Subcommand("measure") {
        val classifier = new Subcommand("classifier") {
          val model: ScallopOption[String] = opt[String]("model", descr = "Available Classifiers:  SVM, Knn, Decision Tree (DT), NaiveBayes (NB)",
            validate = { x => x == "SVM" || x == "KNN" || x == "DT" || x == "NB" })
        }
        addSubcommand(classifier)
        val other: ScallopOption[String] = opt[String]("other", descr = "Available Metrics: F1", validate = { x => x == "F1" || x == "F2" })

      }
      addSubcommand(compMeasure)
      val alpha: ScallopOption[Double] = opt[Double]("alpha", descr = "Aplha Value for threshold computation / Default 0.75", validate = { x => 0 <= x && x <= 1 }, default = Some(0.75))
      verify()
    }

    val globalCompyMeasure = opts.compMeasure.other.getOrElse("None") match {
      case "F1" => fisherRatio _
      case "F2" => f2 _
      case _ => zeroGlobal _
    }

    val filter = opts.feature_algorithm.getOrElse("None")

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
    selectFeatures(opts.dataset(), opts.test_dataset.toOption, opts.class_index(), opts.numParts(), opts.partType(), opts.overlap(), opts.alpha(), globalCompyMeasure, classifier, filter)

  }

  def selectFeatures(dataset_file: String, dataset_test: Option[String], class_is_first: Boolean, numParts: Int, vertical: Boolean, overlap: Double, alpha_value: Double,
                     globalComplexityMeasure: (DataFrame, Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], SparkContext, Option[RDD[(Int, Seq[Any])]], Int) => Double,
                     classifier: Option[PipelineStage], filter: String): Unit = {
    val init_time = System.currentTimeMillis()
    val ss = SparkSession.builder().appName("distributed_feature_selection").master("local[*]").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")
    var dataframe = ss.read.option("maxColumns", "30000").csv(dataset_file)
    var test_dataframe = dataframe

    // If there is not training set, we split the data maintaining class distribution. 2/3 train 1/3 test
    if (dataset_test.isEmpty) {
      println("Splitting train/test")
      val partitioned = dataframe.rdd.map(row => (row.get(row.length - 1), row)).groupByKey()
        .flatMap({
          // Add an index for each subset (keys)
          case (_, value) => value.zipWithIndex
        })
        .map({
          // Get the partition number for each row and make it the new key
          case (row, index) => (index % 3, row)
        })

      val train_set = partitioned.filter(x => x._1 == 1 || x._1 == 2).map(_._2)
      val test_set = partitioned.filter(_._1 == 0).map(_._2)
      dataframe = ss.createDataFrame(train_set, dataframe.schema)
      test_dataframe = ss.createDataFrame(test_set, dataframe.schema)

    } else {

      test_dataframe = ss.read.option("maxColumns", "30000").csv(dataset_test.get)
    }

    println(s"Train: ${dataframe.count()} Test: ${test_dataframe.count()} Time: ${System.currentTimeMillis() - init_time}\n")

    /** *****************************
      * Creation of attributes maps
      * *****************************/

    val map_time = System.currentTimeMillis()
    val class_index = if (class_is_first) 0 else dataframe.columns.length - 1

    //Map creation of attributes
    val inverse_attributes = dataframe.columns.zipWithIndex.map { case (column_name, index) => column_name -> index }.toMap
    val br_inverse_attributes = ss.sparkContext.broadcast(inverse_attributes)

    //Now we have to deal with categorical values
    val first_row = dataframe.first().toSeq.map(_.toString)

    val categorical_filter = ss.sparkContext.parallelize(first_row.zip(dataframe.columns)).filter {
      case (value: String, c_name) =>
        parseNumeric(value).isEmpty || br_inverse_attributes.value(c_name) == class_index
    }.map(_._2).collect()

    val categorical_attributes = dataframe.select(categorical_filter.map { c =>
      collect_set(c)
    }: _*).first().toSeq.zip(categorical_filter).map { case (values: mutable.WrappedArray[String], column_name) => inverse_attributes(column_name) -> (Some(values), column_name) }


    val numerical_attributes = inverse_attributes.keySet.diff(categorical_attributes.map(_._2._2).toSet)
      .map(c_name => inverse_attributes(c_name) -> (None: Option[mutable.WrappedArray[String]], c_name))

    //Finally we add categorical and numerical values
    val attributes = (categorical_attributes ++ numerical_attributes).toMap

    val br_attributes = ss.sparkContext.broadcast(attributes)
    println(s"Attributes in ${System.currentTimeMillis() - map_time}\n")

    /** **************************
      * Getting the Votes vector.
      * **************************/

    val rounds = 5
    val start_time = System.currentTimeMillis()
    if (vertical) println(s"*****Using vertical partitioning with ${overlap * 100}% overlap*****\n*****Using $filter algorithm*****")
    else println(s"*****Using horizontal partitioning*****\n*****Using $filter algorithm*****")

    var transpose_input: Option[RDD[(Int, Seq[Any])]] = None

    val votes = {
      var sub_votes = ss.sparkContext.emptyRDD[(String, Int)]
      if (vertical) {
        transpose_input = Some(transposeRDD(dataframe.rdd))
        for (round <- 1 to rounds) {
          val start_round = System.currentTimeMillis()
          println(s"Round: $round")
          val result = verticalPartitioningFeatureSelection(ss.sparkContext, shuffleRDD(transpose_input.get),
            br_attributes, br_inverse_attributes, class_index, numParts, filter, overlap)
          sub_votes = sub_votes ++ result.map(x => (x._1, x._2._1))
          println(s"Round finished in: ${System.currentTimeMillis() - start_round}. Max computing time is ${result.map(_._2._2).max}")
        }
      } else {
        for (round <- 1 to rounds) {
          val start_round = System.currentTimeMillis()
          println(s"Round: $round")
          val result = horizontalPartitioningFeatureSelection(ss.sparkContext, shuffleRDD(dataframe.rdd),
            br_attributes, br_inverse_attributes, class_index, numParts, filter)
          sub_votes = sub_votes ++ result.map(x => (x._1, x._2._1))
          println(s"Round finished in: ${System.currentTimeMillis() - start_round}. Max computing time is ${result.map(_._2._2).max}")

        }
      }
      sub_votes.reduceByKey(_ + _)
    }

    println(s"Votes computation time:${System.currentTimeMillis() - start_time}\n")
    println(votes.sortBy(_._2).first())

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
    val RDD_inverse_attributes = ss.sparkContext.parallelize(inverse_attributes.toSeq)
    val selected_features_0_votes = RDD_inverse_attributes.map(_._1).subtract(votes.map(_._1))

    //val selected_features_0_votes = inverse_attributes.keySet.diff(votes.map(_._1).toSeq)

    val alpha = alpha_value
    var e_v = collection.mutable.ArrayBuffer[(Int, Double)]()

    val start_comp = System.currentTimeMillis()
    var compMeasure = globalComplexityMeasure(dataframe, br_attributes, ss.sparkContext, transpose_input, class_index)
    println(s"Complexity Measurey Computation Time: ${System.currentTimeMillis() - start_comp}. Value $compMeasure")


    val step = if (vertical) 1 else 5
    for (a <- minVote to maxVote by step) {
      val starting_time = System.currentTimeMillis()
      // We add votes below Threshold value
      val selected_features = selected_features_0_votes ++ votes.filter(_._2 < a).map(_._1)

      if (selected_features.count() > 1) {
        println(s"Starting threshold computation with minVotes = $a / maxVotes = $maxVote with ${selected_features.count() - 1} features")
        val selected_features_dataframe = dataframe.select(selected_features.collect().map(col): _*)
        val retained_feat_percent = (selected_features.count().toDouble / dataframe.columns.length) * 100
        if (classifier.isDefined) {
          val (pipeline_stages, columns_to_cast) = createPipeline(ss.sparkContext.parallelize(selected_features_dataframe.columns), br_attributes, br_inverse_attributes, class_index, ss.sparkContext)
          val casted_dataframe = castDFToDouble(selected_features_dataframe, columns_to_cast)
          val pipeline = new Pipeline().setStages(pipeline_stages :+ classifier.get).fit(casted_dataframe)
          val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
            .setPredictionCol("prediction").setMetricName("accuracy")
          compMeasure = 1 - evaluator.evaluate(pipeline.transform(casted_dataframe))
        }

        e_v += ((a, alpha * compMeasure + (1 - alpha) * retained_feat_percent))

        println(s"\tThreshold computation in ${System.currentTimeMillis() - starting_time} " +
          s"\n\t\t Error: $compMeasure \n\t\t Retained Features Percent: $retained_feat_percent " +
          s"\n\t\t EV Value = ${alpha * compMeasure + (1 - alpha) * retained_feat_percent} \n")

      }
    }

    val selected_threshold = e_v.minBy(_._2)._1
    val features = selected_features_0_votes ++ votes.filter(_._2 < selected_threshold).map(_._1)

    println(s"Total threshold computation in ${System.currentTimeMillis() - threshold_time}")
    println(s"Total execution time is ${System.currentTimeMillis() - init_time}\n")

    /** ******************************************
      * Evaluate Models With Selected Features
      * ******************************************/

    val evaluation_time = System.currentTimeMillis()
    //Once we get the votes, we proceed to evaluate

    val (pipeline_stages, columns_to_cast) = createPipeline(features, br_attributes, br_inverse_attributes, class_index, ss.sparkContext)

    val selected_features_train_dataframe = dataframe.select(features.collect().map(col): _*)
    val selected_features_test_dataframe = test_dataframe.select(features.collect().map(col): _*)

    val casted_train_dataframe = castDFToDouble(selected_features_train_dataframe, columns_to_cast)
    val casted_test_dataframe = castDFToDouble(selected_features_test_dataframe, columns_to_cast)

    val transformation_pipeline = new Pipeline().setStages(pipeline_stages).fit(casted_train_dataframe)
    val transformed_train_dataset = transformation_pipeline.transform(casted_train_dataframe)
    val transformed_test_dataset = transformation_pipeline.transform(casted_test_dataframe)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
      .setPredictionCol("prediction").setMetricName("accuracy")

    println(s"Number of features is ${features.count() - 1}")
    Seq(("SMV", new OneVsRest().setClassifier(new LinearSVC())), ("Decision Tree", new DecisionTreeClassifier()),
      ("Naive Bayes", new NaiveBayes()), ("KNN", new KNNClassifier().setK(1)))
      .foreach {

        case (name, classi) =>

          val accuracy = evaluator.evaluate(classi.fit(transformed_train_dataset).transform(transformed_test_dataset))
          println(s"Accuracy for $name is $accuracy")

      }


    //      #For use with Weka library
    //      val selected_inverse_features_map = inverse_attributes.filterKeys(selected_features.contains(_))
    //      val selected_features_map = attributes.filterKeys(selected_inverse_features_map.values.toSeq.contains(_))
    //      WekaWrapper.createInstances(df, selected_features_map, selected_inverse_features_map, class_index)

    println(s"Evaluation time is ${System.currentTimeMillis() - evaluation_time}\n")
  }


  /** ***********************
    * Partitioning functions
    * ************************/

  def horizontalPartitioningFeatureSelection(sc: SparkContext, input: RDD[Row],
                                             br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], br_inverse_attributes: Broadcast[Map[String, Int]],
                                             class_index: Int, numParts: Int, filter: String): RDD[(String, (Int, Long))] = {

    /** Horizontally partition selection features */


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
      val start_time = System.currentTimeMillis()
      val data = WekaWrapper.createInstances(iter, br_attributes.value, class_index)
      val filtered_data = Filter.useFilter(data, WekaWrapper.filterAttributes(data, filter))
      val selected_attributes = WekaWrapper.getAttributes(filtered_data)

      // Getting the diff we can obtain the features to increase the votes and taking away the class
      (br_inverse_attributes.value.keySet.diff(selected_attributes) - br_attributes.value(class_index)._2).map((_, (1, System.currentTimeMillis() - start_time)))

    }.reduceByKey((t1, t2) => (t1._1 + t2._1, math.max(t1._2, t2._2)))

  }

  def verticalPartitioningFeatureSelection(sc: SparkContext, transposed: RDD[(Int, Seq[Any])],
                                           br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], br_inverse_attributes: Broadcast[Map[String, Int]],
                                           class_index: Int, numParts: Int, filter: String, overlap: Double = 0): RDD[(String, (Int, Long))] = {

    /** Vertically partition selection features */
    val classes = br_attributes.value(class_index)._1.get
    val br_classes = sc.broadcast(classes)

    val overlapping = transposed.sample(withReplacement = false, overlap)
    val br_overlapping = sc.broadcast(overlapping.collect().toIterable)

    // Get the class column
    val br_class_column = sc.broadcast(transposed.filter { case (columnindex, _) => columnindex == class_index }.first())

    //Remove the class column and assign a partition to each column
    transposed.subtract(overlapping).filter { case (columnindex, _) => columnindex != class_index }
      .zipWithIndex.map(line => (line._2 % numParts, line._1))
      .groupByKey().flatMap {
      case (_, iter) =>
        val start_time = System.currentTimeMillis()
        val data = WekaWrapper.createInstancesFromTranspose(iter ++ br_overlapping.value, br_attributes.value, br_class_column.value, br_classes.value)
        val filtered_data = Filter.useFilter(data, WekaWrapper.filterAttributes(data, filter))
        val selected_attributes = WekaWrapper.getAttributes(filtered_data)

        // Getting the diff we can obtain the features to increase the votes and taking away the class
        (br_inverse_attributes.value.keySet.diff(selected_attributes) - br_attributes.value(class_index)._2).map((_, (1, System.currentTimeMillis() - start_time)))


    }.reduceByKey((t1, t2) => (t1._1 + t2._1, math.max(t1._2, t2._2)))


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


  def castDFToDouble(df: DataFrame, columns: Array[String]): DataFrame = {
    //Cast to double of columns that aren't categorical
    df.select(df.columns.map { c =>
      if (columns.contains(c)) {
        col(c).cast("Double")
      } else {
        col(c)
      }
    }: _*)

  }

  def createPipeline(columns: RDD[String],
                     attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]],
                     inverse_attributes: Broadcast[Map[String, Int]],
                     class_index: Int, sc: SparkContext
                    ): (Array[PipelineStage], Array[String]) = {


    //Mllib needs two special columns [features] and [label] to work. We have to assemble the columns we selected
    val double_columns_to_assemble: RDD[String] = columns.filter {
      cname =>
        val original_attr_index = inverse_attributes.value(cname)
        attributes.value(original_attr_index)._1.isEmpty && original_attr_index != class_index
    }

    // Transform categorical data to one_hot in order to work with MLlib
    val stages_columns = columns.filter {
      cname =>
        val original_attr_index = inverse_attributes.value(cname)
        attributes.value(original_attr_index)._1.isDefined && original_attr_index != class_index
    }.map {
      cname =>
        val st_indexer = new StringIndexer().setInputCol(cname).setOutputCol(s"${cname}_index")
        (Array(st_indexer, new OneHotEncoder().setInputCol(st_indexer.getOutputCol).setOutputCol(s"${cname}_vect")), Array(s"${cname}_vect"))
    }.reduce((tuple1, tuple2) => (tuple1._1 ++ tuple2._1, tuple1._2 ++ tuple2._2))

    val columns_to_assemble: Array[String] = double_columns_to_assemble.collect() ++ stages_columns._2
    //Creation of pipeline // Transform class column from categorical to index //Assemble features
    val pipeline: Array[PipelineStage] = stages_columns._1 ++
      Array(new StringIndexer().setInputCol(attributes.value(class_index)._2).setOutputCol("label"), new VectorAssembler().setInputCols(columns_to_assemble).setOutputCol("features")  )

    (pipeline, columns_to_assemble)

  }

  /** ********************
    * Complexity measures
    * *********************/


  def zeroGlobal(dataframe: DataFrame, br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], sc: SparkContext, transposedRDD: Option[RDD[(Int, Seq[Any])]], class_index: Int): Double = {
    0.0
  }

  def fisherRatio(dataframe: DataFrame, br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], sc: SparkContext, transposedRDD: Option[RDD[(Int, Seq[Any])]], class_index: Int): Double = {


    // ProportionclassMap => Class -> Proportion of class
    val samples = dataframe.count().toDouble
    val br_proportionClassMap = sc.broadcast(dataframe.groupBy(dataframe.columns(class_index)).count().rdd.map(row => row(0) -> (row(1).asInstanceOf[Long] / samples.toDouble)).collect().toMap)
    val class_column = sc.broadcast(dataframe.select(dataframe.columns(class_index)).rdd.map(_ (0)).collect())


    val rdd = if (transposedRDD.isDefined) transposedRDD.get else transposeRDD(dataframe.drop(dataframe.columns(class_index)).rdd)
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

  def f2(dataframe: DataFrame, br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], sc: SparkContext, transposedRDD: Option[RDD[(Int, Seq[Any])]], class_index: Int): Double = {

    val class_column = sc.broadcast(dataframe.select(dataframe.columns(class_index)).rdd.map(_ (0)).collect())
    val rdd = if (transposedRDD.isDefined) transposedRDD.get else transposeRDD(dataframe.drop(dataframe.columns(class_index)).rdd)

    var result = 0.0

    val f2 = rdd.map {
      case (column_index, row) =>
        val zipped_row = row.zip(class_column.value)

        //Auxiliar Arrays
        val computed_classes = collection.mutable.ArrayBuffer.empty[String]

        if (br_attributes.value(column_index)._1.isDefined) {
          //If we have categorical values we need to discretize them.
          // We use zipWithIndex where its index is its discretize value.
          val values = br_attributes.value(column_index)._1.get.zipWithIndex.map { case (value, sub_index) => value -> (sub_index + 1) }.toMap

          br_attributes.value(class_index)._1.get.foreach { _class_ =>

            val datasetC = zipped_row.filter(_._2 == _class_).map(x => values(x._2.toString))
            computed_classes += _class_.toString

            br_attributes.value(class_index)._1.get.foreach { sub_class_ =>
              if (!computed_classes.contains(sub_class_)) {
                val datasetK = zipped_row.filter(_._2 == sub_class_).map(x => values(x._2.toString))

                val minmaxi = Seq(datasetC.max, datasetK.max).min
                val maxmini = Seq(datasetC.min, datasetK.min).max
                val maxmaxi = Seq(datasetC.max, datasetK.max).max
                val minmini = Seq(datasetC.min, datasetK.min).min

                result += Seq(0, minmaxi - maxmini).max / (maxmaxi - minmini)
              }
            }

          }
        } else {
          br_attributes.value(class_index)._1.get.foreach { _class_ =>
            val datasetC = zipped_row.filter(_._2 == _class_).map(_._1.toString.toDouble)
            br_attributes.value(class_index)._1.get.foreach { sub_class_ =>
              if (!computed_classes.contains(sub_class_)) {
                val datasetK = zipped_row.filter(_._2 == sub_class_).map(_._1.toString.toDouble)

                val minmaxi = Seq(datasetC.max, datasetK.max).min
                val maxmini = Seq(datasetC.min, datasetK.min).max
                val maxmaxi = Seq(datasetC.max, datasetK.max).max
                val minmini = Seq(datasetC.min, datasetK.min).min

                result += Seq(0, minmaxi - maxmini).max / (maxmaxi - minmini)
              }
            }
          }
        }
    }

    result
  }
}


