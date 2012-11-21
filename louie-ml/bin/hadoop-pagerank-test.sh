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
hadoop fs -rmr /user/louie/fishisland1/output/pagerank >& /dev/null || true
hadoop fs -rmr /user/louie/fishisland1/output/temp >& /dev/null || true

hadoop jar ./target/louie-ml-0.1.0.jar org.louie.ml.graph.pagerank.RandomWalkWithRestartJob \
-libjars /home1/louie09/tc/louie-ml/hadoop-lib/mahout-core-0.8-SNAPSHOT-job.jar \
-Dmapred.child.java.opts="-Xmx1024m" \
--vertices "/user/louie/fishisland1/output/graph/vertices.txt" \
--edges "/user/louie/fishisland1/output/graph/edges.txt" \
--output "/user/louie/fishisland1/output/pagerank" \
--tempDir "/user/louie/fishisland1/output/temp" \
--numIterations "3" \
--dampingFactor "0.85" \
--vertexValueIndex "1";
