package onlyRDD

import java.util

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import org.rogach.scallop.{ScallopConf, ScallopOption}
import weka.core.{Attribute, Instances}
import weka.filters.Filter

import scala.collection.mutable
import scala.reflect.ClassTag


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
    sc.setLogLevel("ERROR")

    val (train_rdd, test_rdd) = createRDDs(opts.dataset(), None, opts.class_index(), sc)
    val attributes = createAttributesMap(train_rdd, sc)
    val br_attributes = sc.broadcast(attributes)

    val fs_algorithms = opts.fs_algorithms().split(",")
    val comp_measures = opts.complexity_measure().split(",")

    val transposed_rdd = if (!opts.partType()) sc.emptyRDD[(Int, Seq[String])] else transposeRDD(train_rdd)
    transposed_rdd.cache()

    var cfs_features_selected = 1

    val (votes, times, cfs_selected) = getVotesVector(train_rdd, br_attributes, opts.numParts(), opts.partType(), opts.overlap(), "CFS", cfs_features_selected, transposed_rdd, sc)
    votes.foreach(print)


  }

  def createRDDs(train_file: String, test_file: Option[String], class_first_column: Boolean, sc: SparkContext): (RDD[Array[String]], RDD[Array[String]]) = {

    def parse_RDD(rdd: RDD[String], sep: Char): RDD[Array[String]] = rdd.map((x: String) => {
      if (class_first_column) {
        val result = x.split(sep)
        result.drop(1) :+ result(0)
      } else x.split(sep)
    })

    var train_rdd = parse_RDD(sc.textFile(train_file), ',')
    var test_rdd = sc.emptyRDD[Array[String]]

    if (test_file.isDefined) {
      test_rdd = parse_RDD(sc.textFile(test_file.get), ',')
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

  def createAttributesMap(rdd: RDD[Array[String]], sc: SparkContext): Map[Int, (Option[Set[String]], String)] = {

    /** *****************************
      * Creation of attributes maps
      * *****************************/

    val sample = rdd.takeSample(false, 1)(0).zipWithIndex
    val nominalAttributes = sample.dropRight(1).filter(tuple => parseNumeric(tuple._1).isEmpty).map(_._2) :+ (sample.length - 1)

    val uniques_nominal_values = rdd.flatMap(_.zipWithIndex).filter { case (_, index) => nominalAttributes.contains(index) }.map(tuple => (tuple._2, tuple._1))
      .combineByKey((value: String) => mutable.Set[String](value),
        (set: mutable.Set[String], new_value: String) => set += new_value,
        (set1: mutable.Set[String], set2: mutable.Set[String]) => set1 ++= set2).map(x => (x._1, Some(x._2.toSet))).collectAsMap()

    sample.map(tuple => {
      if (tuple._2 == (sample.length - 1)) tuple._2 -> (uniques_nominal_values.getOrElse(tuple._2, None), "class")
      else tuple._2 -> (uniques_nominal_values.getOrElse(tuple._2, None), "att_" + tuple._2.toString)
    }).toMap
  }

  def getVotesVector(rdd: RDD[Array[String]], br_attributes: Broadcast[Map[Int, (Option[Set[String]], String)]],
                     numParts: Int, vertical: Boolean, overlap: Double, filter: String, ranking_features: Int,
                     transpose_input: RDD[(Int, Seq[String])],
                     sc: SparkContext): (RDD[(String, Int)], RDD[Long], RDD[Int]) = {

    /** **************************
      * Getting the Votes vector.
      * **************************/

    val rounds = 5
    var times = sc.emptyRDD[Long]
    var cfs_selected = sc.emptyRDD[Int]


    val votes = {
      var sub_votes = sc.emptyRDD[(String, Int)]
      if (vertical) {
        // Get the class column
        val br_class_column = sc.broadcast(transpose_input.filter { case (columnindex, row) => columnindex == br_attributes.value.size - 1 }.first())
        val rdd_no_class_column = transpose_input.filter { case (columnindex, row) => columnindex != br_attributes.value.size - 1 }
        val br_classes = sc.broadcast(br_attributes.value(br_attributes.value.size - 1)._1.get)
        for (_ <- 1 to rounds) {
          val result = verticalPartitioningFeatureSelection(sc, shuffleRDD(rdd_no_class_column),
            br_class_column, br_classes, br_attributes, numParts, filter, overlap, ranking_features)
          sub_votes = sub_votes ++ result.map(x => (x._1, x._2._1))
          times = times ++ result.map(x => x._2._2)
          if (filter == "CFS") {
            cfs_selected = cfs_selected ++ result.map(x => x._2._3)
          }
        }
      } else {
        val (schema, class_schema_index) = WekaWrapperRDD.attributesSchema(br_attributes.value)
        val br_schema = sc.broadcast(schema)
        for (_ <- 1 to rounds) {
          val result = horizontalPartitioningFeatureSelection(sc, shuffleRDD(rdd),
            br_attributes, numParts, filter, br_schema, class_schema_index, ranking_features)
          sub_votes = sub_votes ++ result.map(x => (x._1, x._2._1))
          times = times ++ result.map(x => x._2._2)
          if (filter == "CFS") {
            cfs_selected = cfs_selected ++ result.map(x => x._2._3)
          }
        }
      }
      sub_votes.reduceByKey(_ + _)
    }
    (votes, times, cfs_selected)
  }

  /** ***********************
    * Partitioning functions
    * ************************/

  def horizontalPartitioningFeatureSelection(sc: SparkContext, input: RDD[Array[String]],
                                             br_attributes: Broadcast[Map[Int, (Option[Set[String]], String)]],
                                             numParts: Int, filter: String,
                                             br_attributes_schema: Broadcast[util.ArrayList[Attribute]],
                                             class_schema_index: Int, ranking_features: Int): RDD[(String, (Int, Long, Int))] = {

    /** Horizontally partition selection features */

    input.map(row => (row.last, row)).groupByKey()
      .flatMap({
        // Add an index for each subset (keys)
        case (_, value) => value.zipWithIndex
      })
      .map {
        // Get the partition number for each row and make it the new key
        case (row: Array[String], index: Int) => (index % numParts, row)
      }
      .combineByKey(
        (row: Array[String]) => {
          val data = new Instances("Rel", br_attributes_schema.value, 0)
          data.setClassIndex(class_schema_index)
          WekaWrapperRDD.addRowToInstances(data, br_attributes.value, br_attributes_schema.value, row)
        },
        (inst: Instances, row: Array[String]) => WekaWrapperRDD.addRowToInstances(inst, br_attributes.value, br_attributes_schema.value, row),
        (inst1: Instances, inst2: Instances) => WekaWrapperRDD.mergeInstances(inst1, inst2)
      )
      .flatMap {
        case (_, inst) =>
          val start_time = System.currentTimeMillis()
          val filtered_data = Filter.useFilter(inst, WekaWrapperRDD.filterAttributes(inst, filter, ranking_features))
          val time = System.currentTimeMillis() - start_time
          val selected_attributes = WekaWrapperRDD.getAttributes(filtered_data)
          (br_attributes.value.values.map(_._2).toSet.diff(selected_attributes) - br_attributes.value(br_attributes.value.size - 1)._2).map((_, (1, time, filtered_data.numAttributes())))

      }.reduceByKey((t1, t2) => (t1._1 + t2._1, math.max(t1._2, t2._2), math.max(t1._3, t2._3)))


  }


  def verticalPartitioningFeatureSelection(sc: SparkContext, transposed: RDD[(Int, Seq[String])], br_class_column: Broadcast[(Int, Seq[String])], br_classes: Broadcast[Set[String]],
                                           br_attributes: Broadcast[Map[Int, (Option[Set[String]], String)]],
                                           numParts: Int, filter: String, overlap: Double = 0, ranking_features: Int): RDD[(String, (Int, Long, Int))] = {

    var overlapping: RDD[(Int, Seq[String])] = sc.emptyRDD[(Int, Seq[String])]
    if (overlap > 0) {
      overlapping = transposed.sample(withReplacement = false, overlap)
    }
    val br_overlapping = sc.broadcast(overlapping.collect().toIterable)

    //Remove the class column and assign a partition to each column
    transposed.subtract(overlapping)
      .zipWithIndex.map(line => (line._2 % numParts, line._1))
      .groupByKey().flatMap {
      case (_, iter) =>
        val start_time = System.currentTimeMillis()
        val data = WekaWrapperRDD.createInstancesFromTranspose(iter ++ br_overlapping.value, br_attributes.value, br_class_column.value, br_classes.value)
        val filtered_data = Filter.useFilter(data, WekaWrapperRDD.filterAttributes(data, filter, ranking_features))
        val time = System.currentTimeMillis() - start_time
        val selected_attributes = WekaWrapperRDD.getAttributes(filtered_data)

        // Getting the diff we can obtain the features to increase the votes and taking away the class
        (br_attributes.value.values.map(_._2).toSet.diff(selected_attributes) - br_attributes.value(br_attributes.value.size - 1)._2).map((_, (1, time, filtered_data.numAttributes())))


    }.reduceByKey((t1, t2) => (t1._1 + t2._1, math.max(t1._2, t2._2), math.max(t1._3, t2._3)))


  }


  /** *******************
    * Auxiliar Functions
    * ********************/

  def parseNumeric(s: String): Option[Double] = {
    try {
      Some(s.toDouble)
    } catch {
      case _: Exception => None
    }
  }

  def transposeRDD(rdd: RDD[Array[String]]): RDD[(Int, Seq[String])] = {
    val columnAndRow = rdd.zipWithIndex.flatMap {
      case (row, rowIndex) => row.toSeq.zipWithIndex.map {
        case (element, columnIndex) => columnIndex -> (rowIndex, element)
      }
    }
    columnAndRow.groupByKey.sortByKey().map { case (columnIndex, rowIterable) => (columnIndex, rowIterable.toSeq.sortBy(_._1).map(_._2)) }
  }

  def shuffleRDD[B: ClassTag](rdd: RDD[B]): RDD[B] = {
    rdd.mapPartitions(iter => {
      val rng = new scala.util.Random()
      iter.map((rng.nextInt, _))
    }).partitionBy(new HashPartitioner(rdd.partitions.length)).values
  }

}