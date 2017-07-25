import java.util
import java.io.File

import org.apache.spark.sql.{DataFrame, Row}
import weka.core.{Attribute, DenseInstance, Instances}
import weka.core.converters.ArffSaver

import scala.collection.mutable


object WekaWrapper {

  def createInstances(df: DataFrame, attributes: Map[Int, (Option[Seq[String]], String)], inverse_attributes: Map[String, Int], class_index: Int): Instances = {

    //The list of attributes to create the Weka "Instances"
    val attributes_schema = new util.ArrayList[Attribute]()
    var class_data_index = 0
    df.columns.zipWithIndex.foreach { case (c_name, index) =>
      val column_index = inverse_attributes(c_name)
      if (column_index != class_index) {
        val (value, column_name) = attributes(column_index)
        if (value.isDefined) {
          val attribute_values = new util.ArrayList[String]()
          attributes(column_index)._1.get.foreach(attribute_values.add)
          attributes_schema.add(new Attribute(column_name, attribute_values))
        } else {
          attributes_schema.add(new Attribute(column_name))
        }
      } else {
        //Add classes to attributes
        val classValues = new util.ArrayList[String]()
        attributes(class_index)._1.get.foreach(x => classValues.add(x.toString))
        attributes_schema.add(new Attribute("class", classValues))
        class_data_index = index
      }
    }


    // Weka Instances
    val data = new Instances("Rel", attributes_schema, df.columns.length)
    data.setClassIndex(class_data_index)

    // Once we have the Instances structure, we add the data itself
    df.collect().foreach { row =>
      val instance = new DenseInstance(attributes_schema.size())
      df.columns.zipWithIndex.foreach { case (c_name, index) =>
        val column_index = inverse_attributes(c_name)
        if (attributes(column_index)._1.isDefined) {
          instance.setValue(attributes_schema.get(index), row.getAs(c_name).asInstanceOf[String])
        } else {
          instance.setValue(attributes_schema.get(index), row.getAs(c_name).toString.toDouble)
        }

      }
      data.add(instance)
    }
    data
  }


  def createInstances(iter: Iterable[Row], attributes: Map[Int, (Option[Seq[String]], String)], class_index: Int): Instances = {

    //The list of attributes to create the Weka "Instances"
    val attributes_schema = new util.ArrayList[Attribute]()


    // Getting the attributes and add to schema.
    iter.head.toSeq.dropRight(1).zipWithIndex.foreach { case (_, index) =>
      val (value, column_name) = attributes(index)
      if (value.isDefined) {
        val attribute_values = new util.ArrayList[String]()
        attributes(index)._1.get.foreach(attribute_values.add)
        attributes_schema.add(new Attribute(column_name, attribute_values))
      } else {
        attributes_schema.add(new Attribute(column_name))
      }
    }

    //Add classes to attributes
    val classValues = new util.ArrayList[String]()
    attributes(class_index)._1.get.foreach(x => classValues.add(x.toString))
    attributes_schema.add(new Attribute("class", classValues))

    // Weka Instances
    val data = new Instances("Rel", attributes_schema, iter.size)
    data.setClassIndex(attributes_schema.size() - 1)

    // Once we have the Instances structure, we add the data itself
    iter.foreach({ row =>
      val instance = new DenseInstance(attributes_schema.size())
      row.toSeq.zipWithIndex.foreach({ case (value, index) =>
        if (attributes(index)._1.isDefined) {
          instance.setValue(attributes_schema.get(index), value.asInstanceOf[String])
        } else {
          instance.setValue(attributes_schema.get(index), value.toString.toDouble)
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

  def getAttributes(instances: Instances): mutable.Set[String] = {
    val attributes = collection.mutable.Set[String]()
    val enum_attributes = instances.enumerateAttributes()
    while (enum_attributes.hasMoreElements) {
      attributes += enum_attributes.nextElement().name()
    }

    attributes


  }

  def saveInstances(instances: Instances, file_path: String): Unit = {

    val saver = new ArffSaver
    saver.setInstances(instances)
    saver.setFile(new File(file_path))
    saver.writeBatch()

  }


}



