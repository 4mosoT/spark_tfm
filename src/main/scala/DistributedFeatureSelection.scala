import java.util

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, collect_set}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.{HashPartitioner, SparkContext}
import org.rogach.scallop.{ScallopConf, ScallopOption, Subcommand}
import weka.core.{Attribute, Instances}
import weka.filters.Filter

import scala.collection.mutable
import scala.reflect.ClassTag


object DistributedFeatureSelection {


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

    val start_time = System.currentTimeMillis()
    val ss = SparkSession.builder().appName("distributed_feature_selection").master("local[*]").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")

    val (train_dataframe, test_dataframe) = createDataframes(opts.dataset(), opts.test_dataset.toOption, opts.class_index(), ss)
    train_dataframe.cache()

    val map_time = System.currentTimeMillis()
    val (rdd_inverse_attributes, attributes, inverse_attributes, first_row, class_index) = createAttributesMap(train_dataframe, ss.sparkContext)
    rdd_inverse_attributes.cache()
    println(s"Attributes in ${System.currentTimeMillis() - map_time}\n")

    val fs_algorithms = opts.fs_algorithms().split(",")
    val comp_measures = opts.complexity_measure().split(",")


    val aux_rdd = if (!opts.partType()) ss.sparkContext.emptyRDD[(Int, Seq[Any])] else transposeRDD(train_dataframe.rdd)
    aux_rdd.cache()

    var cfs_features_selected = 1

    for (compmeasure <- comp_measures) {
      for (fsa <- fs_algorithms) {

        val start_sub_time = System.currentTimeMillis()

        if (opts.partType()) println(s"*****Using vertical partitioning with ${opts.overlap() * 100}% overlap*****")
        else println(s"*****Using horizontal partitioning*****")
        println(s"*****Using $fsa algorithm with $compmeasure as complexity measure*****")
        println(s"*****Number of partitions: ${opts.numParts()}*****")

        /** Here we get the votes vector **/
        val (votes, times, transpose) = getVotesVector(train_dataframe, class_index, first_row, attributes, inverse_attributes,
          opts.numParts(), opts.partType(), opts.overlap(), fsa, cfs_features_selected, aux_rdd, ss.sparkContext)

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
        val features = computeThreshold(train_dataframe, rdd_inverse_attributes, votes, opts.alpha(), classifier, attributes, inverse_attributes, opts.partType(),
          opts.numParts(), 5, class_index, transpose, globalCompyMeasure, ss.sparkContext)
        println(s"Feature selection computation time is ${System.currentTimeMillis() - start_sub_time}")


        if (fsa == "CFS") {
          cfs_features_selected = features.count().toInt
        }

        /** Here we evaluate several algorithms with the selected features **/
        val evaluation_time = System.currentTimeMillis()
        evaluateFeatures(train_dataframe, test_dataframe, attributes, inverse_attributes, class_index, features, ss.sparkContext)
        println(s"Evaluation time is ${System.currentTimeMillis() - evaluation_time}")
        println(s"Computation time by partition stats: ${times.stats()}")
        println(s"Trainset: ${train_dataframe.count} Testset: ${test_dataframe.count}")
        println("\n\n")

      }


    }
    println(s"Total script time is ${System.currentTimeMillis() - start_time}")

  }


  def createDataframes(dataset_file: String, dataset_test: Option[String], class_is_first: Boolean, ss: SparkSession): (DataFrame, DataFrame) = {

    var dataframe = ss.read.option("maxColumns", "30000").csv(dataset_file)

    if (class_is_first) {
      val reordered_columns: Array[String] = dataframe.columns.drop(1) :+ dataframe.columns(0)
      dataframe = dataframe.select(reordered_columns.head, reordered_columns.tail: _*)
    }

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
      if (class_is_first) {
        val reordered_columns: Array[String] = test_dataframe.columns.drop(1) :+ test_dataframe.columns(0)
        test_dataframe = test_dataframe.select(reordered_columns.head, reordered_columns.tail: _*)
      }
    }
    (dataframe, test_dataframe)
  }


  def createAttributesMap(dataframe: DataFrame, sc: SparkContext): (RDD[(String, Int)], Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], Broadcast[Map[String, Int]], Row, Int) = {

    /** *****************************
      * Creation of attributes maps
      * *****************************/

    val class_index = dataframe.columns.length - 1

    //Map creation of attributes
    val RDD_columns = sc.parallelize(dataframe.columns)
    val RDD_inverse_attributes: RDD[(String, Int)] = RDD_columns.zipWithIndex.map { case (column_name, index) => column_name -> index.toInt }
    val inverse_attributes = RDD_inverse_attributes.collect().toMap
    val br_inverse_attributes = sc.broadcast(inverse_attributes)

    //Now we have to deal with categorical values
    val first_row = dataframe.first()


    val categorical_filter = sc.parallelize(first_row.toSeq.map(_.toString)).zip(RDD_columns).filter {
      case (value: String, c_name) =>
        parseNumeric(value).isEmpty || br_inverse_attributes.value(c_name) == class_index
    }.map(_._2).collect()

    val categorical_attributes = dataframe.select(categorical_filter.map { c =>
      collect_set(c)
    }: _*).first().toSeq.zip(categorical_filter).map { case (values: mutable.WrappedArray[String], column_name) => inverse_attributes(column_name) -> (Some(values), column_name) }

    val numerical_attributes = RDD_inverse_attributes.map(_._1).subtract(sc.parallelize(categorical_attributes).map(_._2._2))
      .map(c_name => br_inverse_attributes.value(c_name) -> (None: Option[mutable.WrappedArray[String]], c_name))


    //Finally we add categorical and numerical values
    val attributes = (categorical_attributes ++ numerical_attributes.collect()).toMap
    val br_attributes = sc.broadcast(attributes)

    (RDD_inverse_attributes, br_attributes, br_inverse_attributes, first_row, class_index)

  }

  def getVotesVector(dataframe: DataFrame, class_index: Int, first_row: Row, br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], br_inverse_attributes: Broadcast[Map[String, Int]],
                     numParts: Int, vertical: Boolean, overlap: Double, filter: String, ranking_features: Int,
                     transpose_input: RDD[(Int, Seq[Any])],
                     sc: SparkContext): (RDD[(String, Int)], RDD[Long], RDD[(Int, Seq[Any])]) = {

    /** **************************
      * Getting the Votes vector.
      * **************************/

    val rounds = 5
    var times = sc.emptyRDD[Long]
    val trans_input = if (transpose_input.isEmpty() && vertical) transposeRDD(dataframe.rdd) else transpose_input

    val votes = {
      var sub_votes = sc.emptyRDD[(String, Int)]
      if (vertical) {

        // Get the class column
        val br_class_column = sc.broadcast(trans_input.filter { case (columnindex, _) => columnindex == class_index }.first())
        for (_ <- 1 to rounds) {
          val result = verticalPartitioningFeatureSelection(sc, shuffleRDD(trans_input),
            br_class_column, br_attributes, br_inverse_attributes, class_index, numParts, filter, overlap, ranking_features)
          sub_votes = sub_votes ++ result.map(x => (x._1, x._2._1))
          times = times ++ result.map(x => x._2._2)
        }
      } else {
        val (schema, class_schema_index) = WekaWrapper.attributesSchema(first_row, br_attributes.value, class_index)
        val br_schema = sc.broadcast(schema)
        for (_ <- 1 to rounds) {
          val result = horizontalPartitioningFeatureSelectionCombiner(sc, shuffleRDD(dataframe.rdd),
            br_attributes, br_inverse_attributes, class_index, numParts, filter, br_schema, class_schema_index, ranking_features)
          sub_votes = sub_votes ++ result.map(x => (x._1, x._2._1))
          times = times ++ result.map(x => x._2._2)

        }
      }
      sub_votes.reduceByKey(_ + _)
    }
    (votes, times, trans_input)
  }

  def computeThreshold(dataframe: DataFrame, RDD_inverse_attributes: RDD[(String, Int)], votes: RDD[(String, Int)], alpha_value: Double, classifier: Option[PipelineStage],
                       br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], br_inverse_attributes: Broadcast[Map[String, Int]],
                       vertical: Boolean, numParts: Int, rounds: Int = 5, class_index: Int, transpose_input: RDD[(Int, Seq[Any])],
                       globalComplexityMeasure: (DataFrame, Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], SparkContext, RDD[(Int, Seq[Any])], Int) => Double,
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
    val selected_features_0_votes = RDD_inverse_attributes.map(_._1).subtract(votes.map(_._1))
    RDD_inverse_attributes.unpersist()

    val alpha = alpha_value
    var e_v = collection.mutable.ArrayBuffer[(Int, Double)]()


    var compMeasure = 0.0
    val step = if (vertical) 1 else 5
    for (a <- minVote to maxVote by step) {
      val starting_time = System.currentTimeMillis()
      // We add votes below Threshold value
      val selected_features = selected_features_0_votes ++ votes.filter(_._2 < a).map(_._1)

      if (selected_features.count() > 1) {
        println(s"\nStarting threshold computation with minVotes = $a / maxVotes = $maxVote with ${selected_features.count() - 1} features")
        val selected_features_dataframe = dataframe.select(selected_features.collect().map(col): _*)
        val retained_feat_percent = (selected_features.count().toDouble / dataframe.columns.length) * 100
        if (classifier.isDefined) {
          val (pipeline_stages, columns_to_cast) = createPipeline(sc.parallelize(selected_features_dataframe.columns), br_attributes, br_inverse_attributes, class_index, sc)
          val casted_dataframe = castDFToDouble(selected_features_dataframe, columns_to_cast)
          val pipeline = new Pipeline().setStages(pipeline_stages :+ classifier.get).fit(casted_dataframe)
          val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
            .setPredictionCol("prediction").setMetricName("accuracy")
          compMeasure = 1 - evaluator.evaluate(pipeline.transform(casted_dataframe))
        } else {
          val start_comp = System.currentTimeMillis()
          compMeasure = globalComplexityMeasure(selected_features_dataframe, br_attributes, sc, transpose_input, class_index)
          println(s"\n\tComplexity Measure Computation Time: ${System.currentTimeMillis() - start_comp}.")
        }

        e_v += ((a, alpha * compMeasure + (1 - alpha) * retained_feat_percent))

        println(s"\tThreshold computation in ${System.currentTimeMillis() - starting_time} " +
          s"\n\t\t Complexity Measure Value: $compMeasure \n\t\t Retained Features Percent: $retained_feat_percent " +
          s"\n\t\t EV Value = ${alpha * compMeasure + (1 - alpha) * retained_feat_percent} \n")

      }
    }

    val selected_threshold = e_v.minBy(_._2)._1
    val features = selected_features_0_votes ++ votes.filter(_._2 < selected_threshold).map(_._1)


    println(s"Total threshold computation in ${System.currentTimeMillis() - threshold_time}\n")
    println(s"Number of features is ${features.count() - 1}")

    features
  }

  def evaluateFeatures(dataframe: DataFrame, test_dataframe: DataFrame,
                       br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], br_inverse_attributes: Broadcast[Map[String, Int]],
                       class_index: Int, features: RDD[String],
                       sc: SparkContext): Unit = {
    /** ******************************************
      * Evaluate Models With Selected Features
      * ******************************************/

    val (pipeline_stages, columns_to_cast) = createPipeline(features, br_attributes, br_inverse_attributes, class_index, sc)
    val features_columns = features.collect().map(col)

    val selected_features_train_dataframe = dataframe.select(features_columns: _*)
    val selected_features_test_dataframe = test_dataframe.select(features_columns: _*)

    dataframe.unpersist()

    val casted_train_dataframe = castDFToDouble(selected_features_train_dataframe, columns_to_cast)
    val casted_test_dataframe = castDFToDouble(selected_features_test_dataframe, columns_to_cast)

    val transformation_pipeline = new Pipeline().setStages(pipeline_stages).fit(casted_train_dataframe)
    val transformed_train_dataset = transformation_pipeline.transform(casted_train_dataframe)
    transformed_train_dataset.cache()
    val transformed_test_dataset = transformation_pipeline.transform(casted_test_dataframe)
    transformed_test_dataset.cache()

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
      .setPredictionCol("prediction").setMetricName("accuracy")


    Seq(("SMV", new OneVsRest().setClassifier(new LinearSVC())), ("Decision Tree", new DecisionTreeClassifier()),
      ("Naive Bayes", new NaiveBayes()), ("KNN", new KNNClassifier().setTopTreeSize(transformed_train_dataset.count().toInt / 500 + 1).setK(1)))
      .foreach {

        case (name, classi) =>

          val accuracy = evaluator.evaluate(classi.fit(transformed_train_dataset).transform(transformed_test_dataset))
          println(s"Accuracy for $name is $accuracy")

      }

  }


  /** ***********************
    * Partitioning functions
    * ************************/

  def horizontalPartitioningFeatureSelectionCombiner(sc: SparkContext, input: RDD[Row],
                                                     br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], br_inverse_attributes: Broadcast[Map[String, Int]],
                                                     class_index: Int, numParts: Int, filter: String,
                                                     br_attributes_schema: Broadcast[util.ArrayList[Attribute]],
                                                     class_schema_index: Int, ranking_features: Int): RDD[(String, (Int, Long))] = {

    /** Horizontally partition selection features */

    val rdd = input.map(row => (row.get(row.length - 1), row)).groupByKey()
      .flatMap({
        // Add an index for each subset (keys)
        case (_, value) => value.zipWithIndex
      })
      .map {
        // Get the partition number for each row and make it the new key
        case (row: Row, index: Int) => (index % numParts, row)
      }
      .combineByKey(
        (row: Row) => {
          val data = new Instances("Rel", br_attributes_schema.value, 0)
          data.setClassIndex(class_schema_index)
          WekaWrapper.addRowToInstances(data, br_attributes.value, br_attributes_schema.value, row)
        },
        (inst: Instances, row: Row) => WekaWrapper.addRowToInstances(inst, br_attributes.value, br_attributes_schema.value, row),
        (inst1: Instances, inst2: Instances) => WekaWrapper.mergeInstances(inst1, inst2)

      )
    rdd.flatMap {
      case (_, inst) =>
        val start_time = System.currentTimeMillis()
        val filtered_data = Filter.useFilter(inst, WekaWrapper.filterAttributes(inst, filter, ranking_features))
        val time = System.currentTimeMillis() - start_time
        val selected_attributes = WekaWrapper.getAttributes(filtered_data)
        (br_inverse_attributes.value.keySet.diff(selected_attributes) - br_attributes.value(class_index)._2).map((_, (1, time)))

    }.reduceByKey((t1, t2) => (t1._1 + t2._1, math.max(t1._2, t2._2)))


  }


  def verticalPartitioningFeatureSelection(sc: SparkContext, transposed: RDD[(Int, Seq[Any])], br_class_column: Broadcast[(Int, Seq[Any])],
                                           br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], br_inverse_attributes: Broadcast[Map[String, Int]],
                                           class_index: Int, numParts: Int, filter: String, overlap: Double = 0, ranking_features: Int): RDD[(String, (Int, Long))] = {

    /** Vertically partition selection features */
    val classes = br_attributes.value(class_index)._1.get
    val br_classes = sc.broadcast(classes)

    var overlapping: RDD[(Int, Seq[Any])] = sc.emptyRDD[(Int, Seq[Any])]
    if (overlap > 0) {
      overlapping = transposed.sample(withReplacement = false, overlap)
    }
    val br_overlapping = sc.broadcast(overlapping.collect().toIterable)

    //Remove the class column and assign a partition to each column
    transposed.subtract(overlapping).filter { case (columnindex, _) => columnindex != class_index }
      .zipWithIndex.map(line => (line._2 % numParts, line._1))
      .groupByKey().flatMap {
      case (_, iter) =>
        val start_time = System.currentTimeMillis()
        val data = WekaWrapper.createInstancesFromTranspose(iter ++ br_overlapping.value, br_attributes.value, br_class_column.value, br_classes.value)
        val filtered_data = Filter.useFilter(data, WekaWrapper.filterAttributes(data, filter, ranking_features))
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
    //    rdd.mapPartitions(iter => {
    //      val rng = new scala.util.Random()
    //      iter.map((rng.nextInt, _))
    //    }).partitionBy(new HashPartitioner(rdd.partitions.length)).values
    rdd.mapPartitions(new scala.util.Random().shuffle(_))
    rdd

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
    val categorical_columns_filter = columns.filter {
      cname =>
        val original_attr_index = inverse_attributes.value(cname)
        attributes.value(original_attr_index)._1.isDefined && original_attr_index != class_index
    }

    if (categorical_columns_filter.count > 1) {
      val stages_columns =
        categorical_columns_filter.map {
          cname =>
            val st_indexer = new StringIndexer().setInputCol(cname).setOutputCol(s"${cname}_index")
            (Array(st_indexer, new OneHotEncoder().setInputCol(st_indexer.getOutputCol).setOutputCol(s"${cname}_vect")), Array(s"${cname}_vect"))
        }.reduce((tuple1, tuple2) => (tuple1._1 ++ tuple2._1, tuple1._2 ++ tuple2._2))

      val columns_to_assemble: Array[String] = double_columns_to_assemble.collect() ++ stages_columns._2
      //Creation of pipeline // Transform class column from categorical to index //Assemble features
      val pipeline: Array[PipelineStage] = stages_columns._1 ++
        Array(new StringIndexer().setInputCol(attributes.value(class_index)._2).setOutputCol("label"), new VectorAssembler().setInputCols(columns_to_assemble).setOutputCol("features"))

      (pipeline, columns_to_assemble)

    } else {

      val columns_to_assemble: Array[String] = double_columns_to_assemble.collect()
      //Creation of pipeline // Transform class column from categorical to index //Assemble features
      val pipeline: Array[PipelineStage] = Array(new StringIndexer().setInputCol(attributes.value(class_index)._2).setOutputCol("label"), new VectorAssembler().setInputCols(columns_to_assemble).setOutputCol("features"))

      (pipeline, columns_to_assemble)

    }


  }

  /** ********************
    * Complexity measures
    * *********************/


  def zeroGlobal(dataframe: DataFrame, br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], sc: SparkContext, transposedRDD: RDD[(Int, Seq[Any])], class_index: Int): Double = {
    0.0
  }

  def fisherRatio(dataframe: DataFrame, br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], sc: SparkContext, transposedRDD: RDD[(Int, Seq[Any])], class_index: Int): Double = {


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

  def f2(dataframe: DataFrame, br_attributes: Broadcast[Map[Int, (Option[mutable.WrappedArray[String]], String)]], sc: SparkContext, transposedRDD: RDD[(Int, Seq[Any])], class_index: Int): Double = {
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
              result += Seq(0, minmaxi - maxmini).max / (maxmaxi - minmini)

            }

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
              result += Seq(0, minmaxi - maxmini).max / (maxmaxi - minmini)
            }
          }
        }
        result
    }
    f2.sum()

  }
}


