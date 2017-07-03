
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import aux_funcs._

object main extends App {

  val sconf = new SparkConf().setAppName("hsplit").setMaster("local[*]")
  val sc = new SparkContext(sconf)
  sc.setLogLevel("ERROR")
  val input = sc.textFile(args(0))


  val classes = get_classes_and_count(input, ",")

  val num_partitions = 8
  val partitioned = input.map(line => (line.split(",").last, line)).partitionBy(new HorizontalPartitioner(num_partitions, classes.keys.toSet))

  for (x <- 0 until num_partitions) {
    println("Partition " + x)
    val partition = partitioned.mapPartitionsWithIndex((index, iter) => if (index == x) iter else Iterator())
    classes.keys.foreach(x => println(x, partition.filter(_._1 == x).count()))
  }
}