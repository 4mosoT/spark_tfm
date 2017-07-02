
import aux_funcs._
import org.apache.spark.sql.{SQLContext, SparkSession}

object main extends App {

  val ss = SparkSession.builder().appName("hsplit").master("local[*]").getOrCreate()
  val sc = ss.sparkContext

  sc.setLogLevel("ERROR")

  val dataframe = ss.read.csv(args(0))

  val input = dataframe.rdd


  val classes = get_classes_and_count(input)
  println(classes)

  val num_partitions = 8
  val partitioned = input.map(row => (row.get(row.size - 1), row)).partitionBy(new HorizontalPartitioner(num_partitions, classes.keys.toSet))

  for (x <- 0 until num_partitions) {
    println("Partition " + x)
    val partition = partitioned.mapPartitionsWithIndex((index, iter) => if (index == x) iter else Iterator())
    classes.keys.foreach(x => println(x, partition.filter(_._1 == x).count()))
  }
}