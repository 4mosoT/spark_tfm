import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import weka.attributeSelection.{CfsSubsetEval, GreedyStepwise}
import weka.filters.Filter
import weka.filters.supervised.attribute.AttributeSelection


object VerticalPartitioning {


  /** Object for vertical partition a RDD
    */

  def main(args: Array[String]): Unit = {

    //TODO: Parse arguments

    val ss = SparkSession.builder().appName("hsplit").master("local[*]").getOrCreate()

    val dataframe = ss.read.option("maxColumns", "30000").csv(args(0))

    val input = dataframe.rdd
    val numParts: Int = 8

    val part_columns = dataframe.columns.dropRight(1).zipWithIndex.map({ case (colum_name, index) => colum_name -> index % numParts }).toMap
    val class_index = dataframe.columns.length - 1


    //TODO: check if categorical or numeric
    var categorical = true
    val attributes = dataframe.columns.zipWithIndex.map({ case (value, index) =>
      // If categorical we need to add the distinct values it can take plus its column name
      if (categorical) {
        index -> (Some(dataframe.select(dataframe.columns(index)).distinct().collect().toSeq.map(_.get(0)).map(_.toString)), dataframe.columns(index))
      } else {
        // If not categorical we only need column name
        index -> (None, dataframe.columns(index))
      }
    }).toMap

    val br_attributes = ss.sparkContext.broadcast(attributes)
    val index_class = dataframe.columns.length - 1
    val classes = attributes(index_class)._1.get
    val br_classes = ss.sparkContext.broadcast(classes)


    val transposed = transposeRDD(input)
    // Get the class column
    val br_class_column = ss.sparkContext.broadcast(transposed.filter { case (columnindex, row) => columnindex == index_class }.first())
    //Remove the class column and assign a partition to each column
    transposed.filter { case (columnindex, row) => columnindex != index_class }
      .zipWithIndex.map(line => (line._2 % numParts, line._1))
      .groupByKey().map {
      case (_, iter) =>

        val data = WekaWrapper.createInstancesFromTranspose(iter, br_attributes.value, br_class_column.value, br_classes.value)

        //Run Weka Filter to FS
        val filter = new AttributeSelection
        val eval = new CfsSubsetEval
        val search = new GreedyStepwise
        search.setSearchBackwards(true)
        filter.setEvaluator(eval)
        filter.setSearch(search)
        filter.setInputFormat(data)
        Filter.useFilter(data, filter).toSummaryString

    }.foreach(println)

  }

  def transposeRDD(rdd: RDD[Row]): RDD[(Int, Seq[Any])] = {
    val columnAndRow = rdd.zipWithIndex.flatMap {
      case (row, rowIndex) => row.toSeq.zipWithIndex.map {
        case (element, columnIndex) => columnIndex -> (rowIndex, element)
      }
    }
    columnAndRow.groupByKey.sortByKey().map { case (columnIndex, rowIterable) => (columnIndex, rowIterable.toSeq.sortBy(_._1).map(_._2)) }

  }

}


