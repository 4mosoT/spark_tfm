name := "spark_feature_selection"

version := "1.0"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.1",
  "org.apache.spark" %% "spark-sql" % "1.6.1",
  "org.apache.spark" %% "spark-mllib" % "1.6.1"
)


libraryDependencies += "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.1"

resolvers += "Spark Packages Repo" at "http://dl.bintray.com/spark-packages/maven"

libraryDependencies += "saurfang" % "spark-knn" % "0.2.0"

libraryDependencies += "org.rogach" %% "scallop" % "3.0.3"

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}


