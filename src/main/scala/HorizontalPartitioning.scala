import org.apache.spark.sql.SparkSession
import weka.attributeSelection.{CfsSubsetEval, GreedyStepwise}
import weka.filters.Filter
import weka.filters.supervised.attribute.AttributeSelection

object HorizontalPartitioning {


  /** Object for horizontally partition a RDD while maintain the
    * classes distribution
    */

  def main(args: Array[String]): Unit = {

    //TODO: Parse arguments

    val ss = SparkSession.builder().appName("hsplit").master("local[*]").getOrCreate()

    val dataframe = ss.read.option("maxColumns", "30000").csv(args(0))

    val input = dataframe.rdd
    val numParts: Int = 1


    val categorical_attributes = dataframe.columns.zipWithIndex.map({ case (value, index) =>
      index -> dataframe.select(dataframe.columns(index)).distinct().collect().toSeq.map(_.get(0))
    }).toMap
    val br_categorical_attributes = ss.sparkContext.broadcast(categorical_attributes)

    val classes = categorical_attributes(dataframe.columns.length - 1)
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

      val data = WekaWrapper.createInstances(iter, br_categorical_attributes.value, br_classes.value)

      //Run Weka Filter to FS
      val filter = new AttributeSelection
      val eval = new CfsSubsetEval
      val search = new GreedyStepwise
      search.setSearchBackwards(true)
      filter.setEvaluator(eval)
      filter.setSearch(search)
      filter.setInputFormat(data)

      Filter.useFilter(data, filter)


    }

    ).take(10).foreach(println)

  }

  //TODO: Candidate to remove. Not used
  def parseNumeric(s: String): Option[Double] = {
    try {
      Some(s.toDouble)
    } catch {
      case e: Exception => None

    }
  }


}
