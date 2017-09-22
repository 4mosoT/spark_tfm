import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.rogach.scallop.{ScallopConf, ScallopOption}


object DistributedFeatureSelectionRDD {


  def main(args: Array[String]): Unit = {

    //Argument parser
    val opts = new ScallopConf(args) {
      banner("\nUsage of this program example: -d connect-4.csv -p 5 CFS,IG,CF  F1,F2 \n\n")
      val dataset: ScallopOption[String] = opt[String]("dataset", required = true, descr = "Dataset to use in CSV format / Class must be last or first column")
      val test_dataset: ScallopOption[String] = opt[String]("test dataset", descr = "Train dataset to use in CSV format / Class must be last or first column")
      val partType: ScallopOption[Boolean] = toggle("vertical", default = Some(false), descrYes = "Vertical partitioning / Default Horizontal")
      val overlap: ScallopOption[Double] = opt[Double]("overlap", default = Some(0.0), descr = "Overlap")
      val class_index: ScallopOption[Boolean] = toggle("first", noshort = true, default = Some(false), descrYes = "Required if class is first column")
      val numParts: ScallopOption[Int] = opt[Int]("partitions", validate = 0 < _, descr = "Num of partitions", required = true)
      val alpha: ScallopOption[Double] = opt[Double]("alpha", descr = "Aplha Value for threshold computation / Default 0.75", validate = { x => 0 <= x && x <= 1 }, default = Some(0.75))
      val fs_algorithms: ScallopOption[String] = opt[String](required = true, default = Some("CFS,IG,RF"), descr = "List of feature selection algorithm")
      val complexity_measure: ScallopOption[String] = opt[String](required = true, default = Some("F1,F2"), descr = "List of complexity measures")
      verify()
    }

    val conf = new SparkConf().setMaster("local[*]").setAppName("Distributed Feature Selection")
    val sc = new SparkContext(conf)

    val (train_rdd, test_rdd) = createRDDs(opts.dataset(), None, opts.class_index(), sc)
    createAttributesMap(train_rdd, sc)


  }

  def createRDDs(train_file: String, test_file: Option[String], class_first_column: Boolean, sc: SparkContext): (RDD[Array[String]], RDD[Array[String]]) = {

    var train_rdd = sc.textFile(train_file).map(x => {
      if (class_first_column) {
        val result = x.split(",")
        result.drop(1) :+ result(0)
      } else {
        x.split(",")
      }
    })

    var test_rdd = sc.emptyRDD[Array[String]]

    if (test_file.isDefined) {
      test_rdd = sc.textFile(test_file.get).map(x => {
        if (class_first_column) {
          val result = x.split(",")
          result.drop(1) :+ result(0)
        } else {
          x.split(",")
        }
      })


    } else {
      val partitioned = train_rdd.map(row => (row.last, row)).groupByKey()
        .flatMap {
          case (_, value) => value.zipWithIndex
        }
        .map {
          case (row, index) => (index % 3, row)
        }
      train_rdd = partitioned.filter(x => x._1 == 1 || x._1 == 2).map(_._2)
      test_rdd = partitioned.filter(_._1 == 0).map(_._2)


    }
    (train_rdd, test_rdd)
  }

  def createAttributesMap(rdd: RDD[Array[String]], sc: SparkContext) = {


    val z = Array.fill(rdd.first().length)(collection.mutable.Set[String]())
    val uniques = rdd.aggregate(z)({(a, b) => a.zip(b).foreach { case (x, y) => x+= y };a},
                                  {(a, b) => a.zip(b).foreach { case (x, y) => x ++= y};a})
    uniques.foreach(println)

  }


  def parseNumeric(s: String): Option[Double] = {
    try {
      Some(s.toDouble)
    } catch {
      case _: Exception => None
    }
  }

}