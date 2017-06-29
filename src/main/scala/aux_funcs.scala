import org.apache.spark.rdd._

/**
  * Created by marco on 6/29/17.
  */
object aux_funcs {

  def get_classes_and_count(rdd: RDD[String], sep: String): scala.collection.mutable.Map[String, Long] ={

    var classes_list_count = scala.collection.mutable.Map[String, Long] ()

    rdd.foreach( line =>{

      val word = line.split(sep).last
      if (classes_list_count.contains(word)){
        classes_list_count(word) = classes_list_count(word) + 1
      }else{
        classes_list_count(word) = 0

      }

    })


    classes_list_count
  }
}


