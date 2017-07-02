
import aux_funcs._
import org.apache.spark.sql.{SparkSession, Row}

object main extends App {

  val ss = SparkSession.builder().appName("hsplit").master("local[*]").getOrCreate()
  val sc = ss.sparkContext

  sc.setLogLevel("ERROR")


  val dataframe = ss.read.option("maxColumns", "30000").csv(args(0))

  val input = dataframe.rdd

  //Particionado vertical (Matriz Traspuesta)
  val samples = dataframe.count()
  println(samples, dataframe.columns.length)
  val trasposed = traspose_rdd(input)
  println(trasposed.filter(_.length == samples).count())


  // Particionado horizontal
//  val classes = get_classes_and_count(input)
//  println(classes)
//
//  val num_partitions = 8
//  val partitioned = input.map(row => (row.get(row.size - 1), row)).partitionBy(new HorizontalPartitioner(num_partitions, classes.keys.toSet))
//
//  for (x <- 0 until num_partitions) {
//    println("Partition " + x)
//    val partition = partitioned.mapPartitionsWithIndex((index, iter) => if (index == x) iter else Iterator())
//    classes.keys.foreach(x => println(x, partition.filter(_._1 == x).count()))
//  }
}