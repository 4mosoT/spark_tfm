import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD

object HorizontalPartitioning {

  /** Object for horizontally partition a RDD while maintain the
    * classes distribution
    */

  def main(args: Array[String]): Unit = {

    val ss = SparkSession.builder().appName("hsplit").master("local[*]").getOrCreate()
    val dataframe = ss.read.option("maxColumns", "30000").csv(args(0))
    val input = dataframe.rdd
    val classes = get_classes_and_count(input)

    val numParts = 8
    val br_numParts = ss.sparkContext.broadcast(numParts)
    val partitioned = input.map(row => (row.get(row.length - 1), row)).groupByKey()
      .flatMap({
        // Add an index for each subset (keys)
        case (_, value) => value.zipWithIndex
      })
      .map({
        // Get the partition number for each row and make it the new key
        case (row, index) => (index % br_numParts.value, row)
      })

    println(partitioned.countByKey())
  }


  def get_classes_and_count(rdd: RDD[Row]): collection.Map[Any, Long] = {
    //Class is last word of row
    rdd.map(row => row.get(row.size - 1)).countByValue()

  }

}
