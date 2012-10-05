package org.louie.ml.graph.pagerank;

import java.io.File;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.FileLineIterable;

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;

/** example from "Mining Massive Datasets" (page 157) */
public final class PageRankJobTest extends AbstractJob {

  //private static final Logger log = LoggerFactory.getLogger(PageRankJobTest.class);
 
  @Override
	public int run(String[] args) throws Exception {

    String verticesFile = "/user/louie/fishisland/output/graph/vertices.txt";
    String edgesFile = "/user/louie/fishisland/output/graph/edges.txt";
    String outputDir = "/user/louie/fishisland/output/pagerank";
    String tempDir = "/user/louie/fishisland/output/temp";

    Configuration conf = new Configuration();
    conf.set("mapred.child.java.opts", "-Xmx1024m");

    PageRankJob pageRank = new PageRankJob();
    pageRank.setConf(conf);
    pageRank.run(new String[] { "--vertices", verticesFile, "--edges", edgesFile,
        "--output", outputDir, "--numIterations", "3", "--stayingProbability", "0.8",
        "--tempDir", tempDir });

    int numVertices = HadoopUtil.readInt(new Path(tempDir, AdjacencyMatrixJob.NUM_VERTICES), conf);

    Map<Integer,Double> rankPerVertex = Maps.newHashMapWithExpectedSize(numVertices);
    for (CharSequence line : new FileLineIterable(new File(outputDir, "part-m-00000"))) {
      String[] tokens = Iterables.toArray(Splitter.on("\t").split(line), String.class);
      rankPerVertex.put(Integer.parseInt(tokens[0]), Double.parseDouble(tokens[1]));
    }
    
    return 0;
  }
  
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new PageRankJobTest(), args);
	}

}

