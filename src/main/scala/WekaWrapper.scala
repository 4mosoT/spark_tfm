import java.util

import org.apache.spark.sql.Row
import weka.core.{Attribute, DenseInstance, Instances}

object WekaWrapper {

  def createInstances(iter: Iterable[Row], categorical_attributes: Map[Int, Seq[Any]], classes: Seq[Any]): Instances = {

    //The list of attributes to create the Weka "Instances"
    val attributes = new util.ArrayList[Attribute]()

    // Getting the first row and iterate over its elements after indexing them to add to attributes
    iter.head.toSeq.dropRight(1).zipWithIndex.foreach({
      case (value, index) =>
        // If the attribute is categorical we have to add the different values it can take
        if (categorical_attributes.keySet.contains(index)) {
          val attribute_values = new util.ArrayList[String]()
          categorical_attributes(index).foreach(x => attribute_values.add(x.asInstanceOf[String]))
          attributes.add(new Attribute("att_" + index, attribute_values))
        } else {
          attributes.add(new Attribute("att_" + index))
        }
    })

    //Add classes to attributes
    val classValues = new util.ArrayList[String]()
    classes.foreach(x => classValues.add(x.toString))
    attributes.add(new Attribute("class", classValues))

    // Weka Instances
    val data = new Instances("Rel", attributes, iter.size)
    data.setClassIndex(attributes.size() - 1)

    // Once we have the Instances structure, we add the data itself
    iter.foreach({ row =>
      val instance = new DenseInstance(attributes.size())
      row.toSeq.zipWithIndex.foreach({ case (value, index) =>
        if (categorical_attributes.keySet.contains(index)) {
          instance.setValue(attributes.get(index), value.asInstanceOf[String])
        } else {
          instance.setValue(attributes.get(index), value.toString.toDouble)
        }
      })
      data.add(instance)
    })

    data
  }


}
