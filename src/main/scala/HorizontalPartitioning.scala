import org.apache.spark.sql.SparkSession
import weka.attributeSelection.{CfsSubsetEval, GreedyStepwise, InfoGainAttributeEval, Ranker}
import weka.filters.Filter
import weka.filters.supervised.attribute.AttributeSelection
//TODO: Merge with VerticalPartitioning
object HorizontalPartitioning {


  /** Object for horizontally partition a RDD while maintain the
    * classes distribution
    */

  def main(args: Array[String]): Unit = {

    //TODO: Parse arguments

    val ss = SparkSession.builder().appName("hsplit").master("local[*]").getOrCreate()

    val dataframe = ss.read.option("maxColumns", "30000").csv(args(0))

    val input = dataframe.rdd
    val numParts: Int = 10

    val class_index = dataframe.columns.length - 1
    val first_row = dataframe.first().toSeq.map(_.toString)
    val attributes = dataframe.columns.zipWithIndex.map({ case (value, index) =>
      // If categorical we need to add the distinct values it can take plus its column name
      if (parseNumeric(first_row(index)).isEmpty || index == class_index) {
        index -> (Some(dataframe.select(dataframe.columns(index)).distinct().collect().toSeq.map(_.get(0)).map(_.toString)), dataframe.columns(index))
      } else {
        // If not categorical we only need column name
        index -> (None, value)
      }
    }).toMap

    val br_attributes = ss.sparkContext.broadcast(attributes)
    val classes = attributes(class_index)._1.get
    val br_classes = ss.sparkContext.broadcast(classes)


    val partitioned = input.map(row => (row.get(row.length - 1), row)).groupByKey()
      .flatMap({
        // Add an index for each subset (keys)
        case (_, value) => value.zipWithIndex
      })
      .map({
        // Get the partition number for each row and make it the new key
        case (row, index) => (index % numParts, row)
      })

    partitioned.groupByKey().map({ case (_, iter) =>

      val data = WekaWrapper.createInstances(iter, br_attributes.value, br_classes.value)

      //Run Weka Filter to FS
      val filter = new AttributeSelection

      //val eval = new CfsSubsetEval
      //val search = new GreedyStepwise
      //search.setSearchBackwards(true)

      val eval = new InfoGainAttributeEval
      val search = new Ranker
      filter.setEvaluator(eval)
      filter.setSearch(search)
      filter.setInputFormat(data)

      Filter.useFilter(data, filter).toSummaryString


    }

    ).foreach(println)

  }

  def parseNumeric(s: String): Option[Double] = {
    try {
      Some(s.toDouble)
    } catch {
      case e: Exception => None

    }
  }


}
