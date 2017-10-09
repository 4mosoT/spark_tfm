import java.io.File
import java.util

import weka.attributeSelection._
import weka.core.converters.ArffSaver
import weka.core.{Attribute, DenseInstance, Instances}
import weka.filters.supervised.attribute.AttributeSelection

import scala.collection.mutable


object WekaWrapper {

  def createInstancesFromTranspose(iter: Iterable[(Int, Seq[String])], attributes: Map[Int, (Option[Set[String]], String)],
                                   class_column: (Int, Seq[String]), classes: Set[String]): Instances = {

    //The list of attributes to create the Weka "Instances"
    val attributes_schema = new util.ArrayList[Attribute]()
    var indexes = collection.mutable.ArrayBuffer[Int]()

    // Getting the attributes and add to schema. Since iter will only contain the columns to this partition
    // we need to iterate over it and get the info from attributes map
    iter.foreach { case (index, _) =>
      val (value, column_name) = attributes(index)
      if (value.isDefined) {
        val attribute_values = new util.ArrayList[String]()
        attributes(index)._1.get.foreach(attribute_values.add)
        attributes_schema.add(new Attribute(column_name, attribute_values))
        indexes += index
      } else {
        indexes += index
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
      val instance = new DenseInstance(attributes_schema.size)
      row.zipWithIndex.foreach({ case (value, index) =>
        if (attributes(indexes(index))._1.isDefined) {
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



   def attributesSchema(attributes: Map[Int, (Option[Set[String]], String)]): (util.ArrayList[Attribute], Int) = {

    //The list of attributes to create the Weka "Instances"
    val attributes_schema = new util.ArrayList[Attribute]()
    var class_data_index = 0

    // Getting the attributes and add to schema.
    attributes.toSeq.sortBy(_._1).foreach {
      case (column_index, (values, column_name)) =>
        if (column_name != "class") {
          if (values.isDefined) {
            val attribute_values = new util.ArrayList[String]()
            values.get.foreach(attribute_values.add)
            attributes_schema.add(new Attribute(column_name, attribute_values))
          } else attributes_schema.add(new Attribute(column_name))
        } else {
          val classValues = new util.ArrayList[String]()
          values.get.foreach(x => classValues.add(x.toString))
          class_data_index = column_index
          attributes_schema.add(new Attribute("class", classValues))
        }
    }
    (attributes_schema, class_data_index)
  }

  def addRowToInstances(data: Instances, attributes: Map[Int, (Option[Set[String]], String)], attributes_schema: util.ArrayList[Attribute], row: Array[String]): Instances = {

    val instance = new DenseInstance(attributes_schema.size())

    row.zipWithIndex.foreach({ case (value, index) =>
      if (attributes(index)._1.isDefined) {
        instance.setValue(attributes_schema.get(index), value.asInstanceOf[String])
      } else {
        instance.setValue(attributes_schema.get(index), value.toString.toDouble)
      }
    })
    data.add(instance)

    data

  }


  def mergeInstances(inst1: Instances, inst2: Instances): Instances = {

    for (index <- 0 until inst1.size()) {
      inst2.add(inst1.instance(index))
    }
    inst2.compactify()
    inst2

  }

  def filterAttributes(data: Instances, algorithm: String, ranking_features: Int): AttributeSelection = {


    val filter = new AttributeSelection

    if (algorithm == "CFS") {
      val eval = new CfsSubsetEval
      val search = new GreedyStepwise
      filter.setEvaluator(eval)
      filter.setSearch(search)
      filter.setInputFormat(data)
    } else {
      val eval2 = if (algorithm == "IG") new InfoGainAttributeEval else new ReliefFAttributeEval
      val search2 = new Ranker()
      search2.setNumToSelect(ranking_features)
      filter.setEvaluator(eval2)
      filter.setSearch(search2)
      filter.setInputFormat(data)
    }
    filter
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



