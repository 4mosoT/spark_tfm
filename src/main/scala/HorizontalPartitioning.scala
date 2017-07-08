import org.apache.spark.sql.SparkSession

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
    val partitioned = input.map(row => (row.get(row.length - 1), row)).groupByKey()
      .flatMap({
        // Add an index for each subset (keys)
        case (_, value) => value.zipWithIndex
      })
      .map({
        // Get the partition number for each row and make it the new key
        case (row, index) => (index % br_numParts.value, row)
      })

    partitioned.take(10).foreach(println)
  }


}
