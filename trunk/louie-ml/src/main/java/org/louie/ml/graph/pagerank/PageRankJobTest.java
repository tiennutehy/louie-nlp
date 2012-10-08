package org.louie.ml.graph.pagerank;

import java.io.File;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;

/** example from "Mining Massive Datasets" (page 157) */
public final class PageRankJobTest extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(PageRankJobTest.class);
 
  @Override
	public int run(String[] args) throws Exception {
  	
    addOutputOption();
    addOption("vertices", null, "a text file containing all vertices of the graph (one per line)", true);
    addOption("edges", null, "edges of the graph", true);
    addOption("output", null, "output file path", true);
    addOption("numIterations", "it", "number of numIterations", String.valueOf(10));
    addOption("stayingProbability", "tp", "probability not to teleport to a random vertex", String.valueOf(0.85));

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Configuration conf = new Configuration();
    setConf(conf);

    ToolRunner.run(conf, new PageRankJob(), new String[] { "--vertices", getOption("vertices"), "--edges", getOption("edges"),
        "--output", getOption("output"), "--numIterations", getOption("numIterations"), "--stayingProbability", getOption("stayingProbability"),
        "--tempDir", getTempPath().toString() });

    int numVertices = HadoopUtil.readInt(new Path(getOption("tempDir"), AdjacencyMatrixJob.NUM_VERTICES), conf);

    Map<Integer,Double> rankPerVertex = Maps.newHashMapWithExpectedSize(numVertices);
    for (CharSequence line : new FileLineIterable(new File(getOption("output"), "part-m-00000"))) {
      String[] tokens = Iterables.toArray(Splitter.on("\t").split(line), String.class);
      rankPerVertex.put(Integer.parseInt(tokens[0]), Double.parseDouble(tokens[1]));
      log.info(tokens[0] + "\t == " + tokens[1]);
    }
    
    return 0;
  }
  
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new PageRankJobTest(), args);
	}

}

