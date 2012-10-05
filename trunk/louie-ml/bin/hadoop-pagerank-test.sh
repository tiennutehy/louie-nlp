#!/bin/sh

##################################
#
# Global setting & arguments
#
##################################

LANG=ko_KR.utf8
cd ..


##################################
#
# Run Map-Reduce job
#
##################################

export HADOOP_USER_CLASSPATH_FIRST="true"
export HADOOP_CLASSPATH=/home1/louie09/tc/louie-ml/hadoop-lib/mahout-core-0.8-SNAPSHOT-job.jar

echo $HADOOP_CLASSPATH

echo "MapReduce job for PageRank test"
hadoop jar ./target/louie-ml-0.0.1-SNAPSHOT.jar org.louie.ml.graph.pagerank.PageRankJobTest \
-libjars /home1/louie09/tc/louie-ml/hadoop-lib/mahout-core-0.8-SNAPSHOT-job.jar
