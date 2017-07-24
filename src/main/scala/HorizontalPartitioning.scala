import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import weka.attributeSelection.{CfsSubsetEval, GreedyStepwise, InfoGainAttributeEval, Ranker}
import weka.filters.Filter
import weka.filters.supervised.attribute.AttributeSelection

import scala.collection.immutable

//TODO: Merge with VerticalPartitioning

object HorizontalPartitioning {


  /** Object for horizontally partition a RDD while maintain the
    * classes distribution
    */

  def main(args: Array[String]): Unit = {

    //TODO: Parse arguments


    val ss = SparkSession.builder().appName("hsplit").master("local[*]").getOrCreate()

    val dataframe = ss.read.option("maxColumns", "30000").csv(args(0))

    val input = dataframe.rdd
    val numParts: Int = 10


    val class_index = dataframe.columns.length - 1
    val first_row = dataframe.first().toSeq.map(_.toString)
    val inverse_attributes = collection.mutable.Map[String, Int]()
    val attributes = dataframe.columns.zipWithIndex.map({ case (value, index) =>
      // If categorical we need to add the distinct values it can take plus its column name
      if (parseNumeric(first_row(index)).isEmpty || index == class_index) {
        inverse_attributes += value -> index
        index -> (Some(dataframe.select(dataframe.columns(index)).distinct().collect().toSeq.map(_.get(0).toString)), value)
      } else {
        // If not categorical we only need column name
        inverse_attributes += value -> index
        index -> (None, value)
      }
    }).toMap

    val br_inverse_attributes = ss.sparkContext.broadcast(inverse_attributes)
    val br_attributes = ss.sparkContext.broadcast(attributes)
    val classes = attributes(class_index)._1.get
    val br_classes = ss.sparkContext.broadcast(classes)

    val start_program_time = System.currentTimeMillis()
    println(fisherRatio(dataframe, br_attributes.value, ss.sparkContext))
    println(System.currentTimeMillis() - start_program_time)

    //    val partitioned = input.map(row => (row.get(row.length - 1), row)).groupByKey()
    //      .flatMap({
    //        // Add an index for each subset (keys)
    //        case (_, value) => value.zipWithIndex
    //      })
    //      .map({
    //        // Get the partition number for each row and make it the new key
    //        case (row, index) => (index % numParts, row)
    //      })
    //
    //    val votes = partitioned.groupByKey().flatMap { case (_, iter) =>
    //
    //      val data = WekaWrapper.createInstances(iter, br_attributes.value, br_classes.value)
    //
    //      //Run Weka Filter to FS
    //      val filter = new AttributeSelection
    //
    //      val eval = new CfsSubsetEval
    //      val search = new GreedyStepwise
    //      search.setSearchBackwards(true)
    //
    //      //val eval = new InfoGainAttributeEval
    //      //val search = new Ranker
    //
    //      filter.setEvaluator(eval)
    //      filter.setSearch(search)
    //      filter.setInputFormat(data)
    //      val filtered_data = Filter.useFilter(data, filter)
    //
    //      val selected_attributes = WekaWrapper.getAttributes(filtered_data)
    //      // Getting the diff we can obtain the features to increase the votes
    //      (br_inverse_attributes.value.keySet.diff(selected_attributes) - br_attributes.value(class_index)._2).map((_, 1))
    //
    //    }.reduceByKey(_ + _).collect()
    //
    //    print(votes.sortBy(_._1).mkString(","))

  }

  def parseNumeric(s: String): Option[Double] = {
    try {
      Some(s.toDouble)
    } catch {
      case e: Exception => None

    }
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