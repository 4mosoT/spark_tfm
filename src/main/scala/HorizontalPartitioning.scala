import java.util

import org.apache.spark.sql.SparkSession

import weka.core.Attribute
import weka.core.{Instances, DenseInstance}

object HorizontalPartitioning {


  /** Object for horizontally partition a RDD while maintain the
    * classes distribution
    */

  def main(args: Array[String]): Unit = {

    //TODO: Parse arguments

    val ss = SparkSession.builder().appName("hsplit").master("local[*]").getOrCreate()

    val dataframe = ss.read.option("maxColumns", "30000").csv(args(0))

    val input = dataframe.rdd
    val numParts = 8
    val br_numParts = ss.sparkContext.broadcast(numParts)

    val classes = dataframe.select(dataframe.columns.last).distinct().collect().toSeq.map(_.get(0))
    val br_classes = ss.sparkContext.broadcast(classes)

    val partitioned = input.map(row => (row.get(row.length - 1), row)).groupByKey()
      .flatMap({
        // Add an index for each subset (keys)
        case (_, value) => value.zipWithIndex
      })
      .map({
        // Get the partition number for each row and make it the new key
        case (row, index) => (index % br_numParts.value, row)
      })

    partitioned.groupByKey().map({ case (_, iter) =>

      val attributes = new util.ArrayList[Attribute]()
      iter.head.toSeq.dropRight(1).zipWithIndex.foreach({ case (value, index) => attributes.add(new Attribute("att_" + index)) })
      val classValues = new util.ArrayList[String]()
      classes.foreach(x => classValues.add(x.toString))
      attributes.add(new Attribute("class", classValues))

      val isTrainingSet = new Instances("Rel", attributes, iter.size)
      isTrainingSet.setClassIndex(attributes.size() - 1)

      val instance = new DenseInstance(attributes.size())

      isTrainingSet


    }).take(10).foreach(println)

  }


}
