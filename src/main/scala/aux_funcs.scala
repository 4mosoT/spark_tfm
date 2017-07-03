import org.apache.spark.rdd.RDD
import org.apache.spark.Partitioner

/**
  * Created by marco on 6/29/17.
  */
object aux_funcs {

  def get_classes_and_count(rdd: RDD[String], sep: String): collection.Map[String, Long] = {
    //Class is last word of row
    rdd.map(_.split(sep).last).countByValue()

  }

  class HorizontalPartitioner(numParts: Int, classes: Set[String]) extends Partitioner {

    private val mapper = classes.map(_ -> 0).toMap
    private val mapper_2 = collection.mutable.Map(mapper.toSeq: _*)


    override def numPartitions: Int = numParts

    override def getPartition(key: Any): Int = {
      val key_count = mapper_2(key.toString)
      mapper_2(key.toString) += 1
      key_count % numParts

    }
  }

}


