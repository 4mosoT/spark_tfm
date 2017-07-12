import java.util
import java.io.File

import org.apache.spark.sql.Row
import weka.core.{Attribute, DenseInstance, Instances}
import weka.core.converters.ArffSaver


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

  def createInstancesFromTranspose(iter: Iterable[(Int, Seq[Any])], attributes: Map[Int, (Option[Seq[String]], String)],
                                   class_column: (Int, Seq[Any]), classes: Seq[String]): Instances = {
    //The list of attributes to create the Weka "Instances"
    val attributes_schema = new util.ArrayList[Attribute]()

    // Getting the attributes and add to schema. Since iter will only contain the columns to this partition
    // we need to iterate over it and get the info from attributes map
    iter.foreach { case (index, _) =>
      val (value, column_name) = attributes(index)
      if (value.isDefined) {
        val attribute_values = new util.ArrayList[String]()
        attributes(index)._1.get.foreach(attribute_values.add)
        attributes_schema.add(new Attribute(column_name, attribute_values))
      } else {
        attributes_schema.add(new Attribute(column_name))
      }
    }

    //Add the classes to schema
    val classValues = new util.ArrayList[String]()
    classes.foreach(x => classValues.add(x.toString))
    attributes_schema.add(new Attribute("class", classValues))

    // Weka Instances
    val data = new Instances("Rel", attributes_schema, iter.size)
    data.setClassIndex(attributes_schema.size - 1)

    //Add the data itself
    val rows = iter.map(_._2).transpose
    rows.foreach({ row =>
      val instance = new DenseInstance(attributes_schema.size + 1)
      row.zipWithIndex.foreach({ case (value, index) =>
        if (attributes(index)._1.isDefined) {
          instance.setValue(attributes_schema.get(index), value.asInstanceOf[String])
        } else {
          instance.setValue(attributes_schema.get(index), value.toString.toDouble)
        }
      })
      data.add(instance)
    })

    // Add class column
    class_column._2.zipWithIndex.foreach { case (value, index) =>
      data.instance(index).setValue(attributes_schema.get(attributes_schema.size - 1), value.toString)
    }

    data


  }

  def saveInstances(instances: Instances, file_path: String): Unit = {

    val saver = new ArffSaver
    saver.setInstances(instances)
    saver.setFile(new File(file_path))
    saver.writeBatch()

  }


}



