#!/usr/bin/env bash

~/spark/bin/spark-submit --driver-memory 5g --executor-memory 5G --master $1  --class DistributedFeatureSelection ~/IdeaProjects/spark_test/target/scala-2.11/spark_feature_selection-assembly-1.0.jar ${*:2}

# -d ~/datasets/connect-4.data -p 10 measure classifier -m SVM
#spar://192.168.1.119:7077