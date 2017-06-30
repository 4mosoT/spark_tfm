import org.apache.spark.sql.SparkSession

object main extends App {

  val sparkSession = SparkSession.builder().appName("ClusterInvoice").master("local[4]").getOrCreate()
  sparkSession.sparkContext.setLogLevel("ERROR")


    val file_path = args(0)
    val df = sparkSession.read.option("maxColumns", 30000).csv(file_path)
    print(df.columns.length, df.count())

    df.rdd.pa

}