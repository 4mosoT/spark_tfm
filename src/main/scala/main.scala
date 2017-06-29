
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import aux_funcs._

object main extends App {

  val sconf = new SparkConf().setAppName("hsplit").setMaster("local[*]")
  val sc = new SparkContext(sconf)
  sc.setLogLevel("ERROR")
  val input = sc.textFile(args(0))

  println(get_classes_and_count(input, ","))










}