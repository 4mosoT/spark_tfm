import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

object main extends App {
  val conf = new SparkConf().setMaster("local").setAppName("spark_test")
  val sc = new SparkContext(conf)
}