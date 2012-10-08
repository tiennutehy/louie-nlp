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

echo "MapReduce job for PageRank......"
hadoop fs -rmr /user/louie/fishisland/output/pagerank >& /dev/null || true
hadoop fs -rmr /user/louie/fishisland/output/temp >& /dev/null || true

hadoop jar ./target/louie-ml-0.0.1-SNAPSHOT.jar org.louie.ml.graph.pagerank.PageRankJob \
-libjars /home1/louie09/tc/louie-ml/hadoop-lib/mahout-core-0.8-SNAPSHOT-job.jar \
-Dmapred.child.java.opts="-Xmx1024m" \
--vertices "/user/louie/fishisland/output/graph/vertices.txt" \
--edges "/user/louie/fishisland/output/graph/edges.txt" \
--output "/user/louie/fishisland/output/pagerank" \
--numIterations "3" \
--stayingProbability "0.8" \
--tempDir "/user/louie/fishisland/output/temp";
