import java.io.File
import java.util

import org.apache.spark.sql.{DataFrame, Row}
import weka.attributeSelection._
import weka.core.converters.ArffSaver
import weka.core.{Attribute, DenseInstance, Instances}
import weka.filters.Filter
import weka.filters.supervised.attribute.AttributeSelection

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

  def attributesSchema(row: Row, attributes: Map[Int, (Option[Seq[String]], String)], class_index: Int): (util.ArrayList[Attribute], Int) = {

    //The list of attributes to create the Weka "Instances"
    val attributes_schema = new util.ArrayList[Attribute]()
    var class_data_index = 0

    // Getting the attributes and add to schema.
    row.toSeq.zipWithIndex.foreach { case (_, index) =>
      val (value, column_name) = attributes(index)
      if (index != class_index) {
        if (value.isDefined) {
          val attribute_values = new util.ArrayList[String]()
          attributes(index)._1.get.foreach(attribute_values.add)
          attributes_schema.add(new Attribute(column_name, attribute_values))
        } else {
          attributes_schema.add(new Attribute(column_name))
        }
      } else {
        val classValues = new util.ArrayList[String]()
        attributes(class_index)._1.get.foreach(x => classValues.add(x.toString))
        attributes_schema.add(new Attribute("class", classValues))
        class_data_index = index

      }
    }

    (attributes_schema, class_data_index)

  }


  def createInstancesFromSchema(iter: Iterable[Row], attributes: Map[Int, (Option[Seq[String]], String)],
                                attributes_schema: util.ArrayList[Attribute], class_data_index: Int): Instances = {

    // Weka Instances
    val data = new Instances("Rel", attributes_schema, iter.size)
    data.setClassIndex(class_data_index)

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


  def createInstances(iter: Iterable[Row], attributes: Map[Int, (Option[Seq[String]], String)], class_index: Int): Instances = {

    val (attributes_schema, class_data_index) = attributesSchema(iter.head, attributes, class_index)

    // Weka Instances
    val data = new Instances("Rel", attributes_schema, 0)
    data.setClassIndex(class_data_index)

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

  def addRowToInstances(data: Instances, attributes: Map[Int, (Option[Seq[String]], String)], attributes_schema: util.ArrayList[Attribute], row: Row): Instances = {

    val instance = new DenseInstance(attributes_schema.size())
    row.toSeq.zipWithIndex.foreach({ case (value, index) =>
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

  def filterAttributes(data: Instances, algorithm: String): AttributeSelection = {

    //We will always run CFS
    var filter = new AttributeSelection
    val eval = new CfsSubsetEval
    val search = new GreedyStepwise
    search.setSearchBackwards(true)
    filter.setEvaluator(eval)
    filter.setSearch(search)
    filter.setInputFormat(data)

    if (algorithm != "CFS") {
      //If not CFS we need the number of attributes CFS selected
      val filtered_data = Filter.useFilter(data, filter)
      val selected_attributes = WekaWrapper.getAttributes(filtered_data)
      filter = new AttributeSelection
      val eval2 = if (algorithm == "IG") new InfoGainAttributeEval else new ReliefFAttributeEval
      val search2 = new Ranker()
      search2.setNumToSelect(selected_attributes.size)
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



