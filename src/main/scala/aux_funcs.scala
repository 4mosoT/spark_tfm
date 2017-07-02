import org.apache.spark.rdd.RDD
import org.apache.spark.Partitioner
import org.apache.spark.sql.Row

/**
  * Created by marco on 6/29/17.
  */
object aux_funcs {

  def get_classes_and_count(rdd: RDD[Row]): collection.Map[Any, Long] = {
    //Class is last word of row
    rdd.map(row => row.get(row.size - 1)).countByValue()

  }

  class HorizontalPartitioner(numParts: Int, classes: Set[Any]) extends Partitioner {

    private val mapper = classes.map(_ -> 0).toMap
    private val mapper_2 = collection.mutable.Map(mapper.toSeq: _*)


    override def numPartitions: Int = numParts

    override def getPartition(key: Any): Int = {
      val key_count = mapper_2(key)
      mapper_2(key) += 1
      key_count % numParts

    }
  }

}


