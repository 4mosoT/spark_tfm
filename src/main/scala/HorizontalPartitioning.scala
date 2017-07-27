import org.apache.spark.SparkContext
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.rogach.scallop.{ScallopConf, ScallopOption, Subcommand}
import weka.attributeSelection.{CfsSubsetEval, GreedyStepwise, InfoGainAttributeEval, Ranker}
import weka.filters.Filter
import weka.filters.supervised.attribute.AttributeSelection


//TODO: Merge with VerticalPartitioning


object HorizontalPartitioning {


  /** Object for horizontally partition a RDD while maintain the
    * classes distribution
    */

  def main(args: Array[String]): Unit = {

    //Argument parser
    //TODO: Parse feature selection algorithm
    val opts = new ScallopConf(args) {
      banner("\nUsage of this program: -d file -p number_of_partitions -v true_if_vertical_partition(default = false) measure (classifier -m classifier | -o another classifier) \n\n" +
        "Examples:  -d connect-4.data -p 10 measure classifier -m SVM \n\t\t   -d connect-4.data -p 10 measure -o F1 \n"
      )
      val dataset: ScallopOption[String] = opt[String]("dataset", required = true, descr = "Dataset to use in CSV format / Class must be last column")
      val partType: ScallopOption[Boolean] = toggle("vertical", default = Some(false), descrYes = "Vertical partitioning / Default Horizontal")
      val numParts = opt[Int]("partitions", validate = 0 <, descr = "Num of partitions", required = true)
      val compMeasure = new Subcommand("measure") {
        val classifier = new Subcommand("classifier") {
          val model = opt[String]("model", descr = "Available Classifiers:  SVM, Knn, C4.5, NaiveBayes (NB)")
        }
        addSubcommand(classifier)
        val other = opt[String]("other", descr = "Metrics available: F1")

      }
      addSubcommand(compMeasure)
      val alpha = opt[Double]("alpha", descr = "Aplha Value for threshold computation / Default 0.75", validate = x => 0 <= x && x <= 1, default = Some(0.75))
      verify()
    }

    val dataset_file = opts.dataset()
    val numParts = opts.numParts()


    val ss = SparkSession.builder().appName("hsplit").master("local[*]").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")
    val dataframe = ss.read.option("maxColumns", "30000").csv(dataset_file)

    val input = dataframe.rdd

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
    val br_inverse_attributes = ss.sparkContext.broadcast(inverse_attributes)
    val br_attributes = ss.sparkContext.broadcast(attributes)


    /** **************************
      * Getting the Votes vector.
      * **************************/

    val partitioned = input.map(row => (row.get(row.length - 1), row)).groupByKey()
      .flatMap({
        // Add an index for each subset (keys)
        case (_, value) => value.zipWithIndex
      })
      .map({
        // Get the partition number for each row and make it the new key
        case (row, index) => (index % numParts, row)
      })

    val votes = partitioned.groupByKey().flatMap { case (_, iter) =>

      val data = WekaWrapper.createInstances(iter, br_attributes.value, class_index)

      //Run Weka Filter to FS
      val filter = new AttributeSelection

      val eval = new CfsSubsetEval
      val search = new GreedyStepwise
      search.setSearchBackwards(true)

      //val eval = new InfoGainAttributeEval
      //val search = new Ranker

      filter.setEvaluator(eval)
      filter.setSearch(search)
      filter.setInputFormat(data)
      val filtered_data = Filter.useFilter(data, filter)

      val selected_attributes = WekaWrapper.getAttributes(filtered_data)
      // Getting the diff we can obtain the features to increase the votes and taking away the class
      (br_inverse_attributes.value.keySet.diff(selected_attributes) - br_attributes.value(class_index)._2).map((_, 1))

    }.reduceByKey(_ + _).collect()

    /** ******************************************
      * Computing 'Selection Features Threshold'
      * ******************************************/

    val avg_votes = votes.map(_._2).sum.toDouble / votes.length
    val std_votes = math.sqrt(votes.map(votes => math.pow(votes._2 - avg_votes, 2)).sum / votes.length)
    val minVote = (avg_votes - (std_votes / 2)).toInt
    val maxVote = (avg_votes + (std_votes / 2)).toInt

    //We get the features that aren't in the votes set. That means features -> Votes = 0
    // **Class column included**
    val selected_features_0_votes = inverse_attributes.keySet.diff(votes.map(_._1).toSet)

    val alpha = opts.alpha()
    var e_v = collection.mutable.ArrayBuffer[(Int, Double)]()

    var compMeasure = if (opts.compMeasure.other.isDefined) opts.compMeasure.other() match {
      case "F1" => fisherRatio(dataframe, br_attributes.value, ss.sparkContext)
    } else 0.0

    val classifier = if (opts.compMeasure.classifier.model.isDefined) opts.compMeasure.classifier.model() match {
      case "SVM" =>
        new OneVsRest().setClassifier(new LinearSVC())
      case "KNN" =>
        new KNNClassifier().setK(1)
      case "C4.5" =>
        new DecisionTreeClassifier()
      case "NB" =>
        new NaiveBayes()
    } else new NaiveBayes()

    for (a <- minVote to maxVote by 1) {
      // We add votes below Threshold value
      val selected_features = (selected_features_0_votes ++ votes.filter(_._2 < a).map(_._1)).toSeq
      val selected_features_dataframe = dataframe.select(selected_features.head, selected_features.tail: _*)
      val retained_feat_percent = (selected_features.length.toDouble / dataframe.columns.length) * 100

      if (opts.compMeasure.classifier.model.isDefined)
        compMeasure = classification_error(selected_features_dataframe, attributes, inverse_attributes, class_index, classifier)


      e_v += ((a, alpha * compMeasure + (1 - alpha) * retained_feat_percent))

    }


    val selected_threshold = e_v.minBy(_._2)._1
    val selected_features_threshold = (selected_features_0_votes ++ votes.filter(_._2 < selected_threshold).map(_._1)) - attributes(class_index)._2
    println(selected_features_threshold)


    //      #For use with Weka library
    //      val selected_inverse_features_map = inverse_attributes.filterKeys(selected_features.contains(_))
    //      val selected_features_map = attributes.filterKeys(selected_inverse_features_map.values.toSeq.contains(_))
    //      WekaWrapper.createInstances(df, selected_features_map, selected_inverse_features_map, class_index)

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
    val df = columns_to_assemble.foldLeft(selected_features_dataframe) { case (new_df, cname) =>
      new_df.withColumn(cname, new_df(cname).cast("double"))
    }

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


  def fisherRatio(dataframe: DataFrame, attributes: Map[Int, (Option[Seq[String]], String)], sc: SparkContext): Double = {

    // ProportionclassMap => Class -> Proportion of class
    val samples = dataframe.count().toDouble
    val br_proportionClassMap = sc.broadcast(dataframe.groupBy(dataframe.columns.last).count().rdd.map(row => row(0) -> (row(1).asInstanceOf[Long] / samples.toDouble)).collect().toMap)

    //Auxiliar Arrays
    val f_feats = collection.mutable.ArrayBuffer.empty[Double]
    val computed_classes = collection.mutable.ArrayBuffer.empty[String]

    dataframe.columns.dropRight(1).zipWithIndex.foreach { case (column_name, index) =>
      var sumMean: Double = 0
      var sumVar: Double = 0

      if (attributes(index)._1.isDefined) {
        //If we have categorical values we need to discretize them.
        // We use zipWithIndex where its index is its discretize value.
        val br_values = sc.broadcast(attributes(index)._1.get.zipWithIndex.map { case (value, index) => value -> (index + 1) }.toMap)

        br_proportionClassMap.value.keySet.foreach { _class_ =>

          val filtered_class_column = dataframe.filter(dataframe(dataframe.columns.last).equalTo(_class_)).select(column_name)
          val mean_class = filtered_class_column.groupBy(column_name).count().rdd.map(row => br_values.value(row.get(0).toString) * row.get(1).asInstanceOf[Long]).reduce(_ + _) / samples.toDouble

          computed_classes += _class_.toString

          br_proportionClassMap.value.keySet.foreach { sub_class_ =>

            if (!computed_classes.contains(sub_class_)) {
              val mean_sub_class = dataframe.filter(dataframe(dataframe.columns.last).equalTo(sub_class_))
                .select(column_name).groupBy(column_name).count().rdd.map(row => br_values.value(row.get(0).toString) * row.get(1).asInstanceOf[Long]).reduce(_ + _) / samples.toDouble

              sumMean += scala.math.pow(mean_class - mean_sub_class, 2) * br_proportionClassMap.value(_class_) * br_proportionClassMap.value(sub_class_)
            }
          }
          val variance = filtered_class_column
            .rdd.map(row => math.pow(br_values.value(row.get(0).toString) - mean_class, 2)).reduce(_ + _) / samples.toDouble
          sumVar += variance * br_proportionClassMap.value(_class_)
        }
        f_feats += sumMean / sumVar

      } else {

        br_proportionClassMap.value.keySet.foreach { _class_ =>

          val filtered_class_column = dataframe.filter(dataframe(dataframe.columns.last).equalTo(_class_)).select(column_name)
          val mean_class = filtered_class_column.rdd.map(_.get(0).toString.toDouble).reduce(_ + _) / samples

          computed_classes += _class_.toString

          br_proportionClassMap.value.keySet.foreach { sub_class_ =>

            if (!computed_classes.contains(sub_class_)) {
              val mean_sub_class = dataframe.filter(dataframe(dataframe.columns.last).equalTo(sub_class_)).rdd
                .map(_.get(0).toString.toDouble).reduce(_ + _) / samples.toDouble

              sumMean += scala.math.pow(mean_class - mean_sub_class, 2) * br_proportionClassMap.value(_class_) * br_proportionClassMap.value(sub_class_)
            }
          }
          val variance = filtered_class_column
            .rdd.map(row => math.pow(row.get(0).toString.toDouble - mean_class, 2)).reduce(_ + _) / samples.toDouble
          sumVar += variance * br_proportionClassMap.value(_class_)
        }
        f_feats += sumMean / sumVar


      }
    }

    1 / f_feats.max

  }


}



