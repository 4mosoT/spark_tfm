import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import weka.attributeSelection.{CfsSubsetEval, GreedyStepwise}
import weka.filters.Filter
import weka.filters.supervised.attribute.AttributeSelection

import scala.collection.mutable.ArrayBuffer

object VerticalPartitioning {


  /** Object for vertical partition a RDD
    */

  def main(args: Array[String]): Unit = {

    //TODO: Parse arguments

    val ss = SparkSession.builder().appName("hsplit").master("local[*]").getOrCreate()

    val dataframe = ss.read.option("maxColumns", "30000").csv(args(0))

    val input = dataframe.rdd
    val numParts: Int = 8

    val part_columns = dataframe.columns.dropRight(1).zipWithIndex.map({ case (colum_name, index) => colum_name -> index % numParts }).toMap
    val class_index = dataframe.columns.length - 1


    //TODO: check if categorical or numeric
    var categorical = true
    val attributes = dataframe.columns.zipWithIndex.map({ case (value, index) =>
      // If categorical we need to add the distinct values it can take plus its partition index
      if (categorical) {
        index -> (Some(dataframe.select(dataframe.columns(index)).distinct().collect().toSeq.map(_.get(0))), index % numParts)
      } else {
        // If not categorical we only need partition index
        index -> (None, index % numParts)
      }
    }).toMap

    val br_attributes = ss.sparkContext.broadcast(attributes)
    val classes = attributes(dataframe.columns.length - 1)
    val br_classes = ss.sparkContext.broadcast(classes)


    val trasposed = trasposeRDD(input)
    val partitioned = trasposed.zipWithIndex.map(line => (line._2 % numParts, line._1))


  }

  def trasposeRDD(rdd: RDD[Row]): RDD[Any] = {
    val columnAndRow = rdd.zipWithIndex.flatMap {
      case (row, rowIndex) => row.toSeq.zipWithIndex.map {
        case (element, columnIndex) => columnIndex -> (rowIndex, element)
      }
    }
    val byColumns = columnAndRow.groupByKey.sortByKey().values
    byColumns.map { indexedRow => indexedRow.toSeq.sortBy(_._1).map(_._2) }


  }

}