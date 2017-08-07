#!/usr/bin/env bash

~/spark/bin/spark-submit --driver-memory 5g --executor-memory 5G --master $1  --class DistributedFeatureSelection $2 ${*:3}

# -d ~/datasets/connect-4.data -p 10 measure classifier -m SVM
#spark://192.168.1.119:7077