package org.louie.ml.graph.pagerank;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * <p>Distributed computation of the proximities of vertices to a source vertex in a directed graph</p>
 *
 * <p>This job outputs text files with a vertex id and its pagerank per line.</p>
  *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--output=(path): output path</li>
 * <li>--vertices=(path): file containing the list of vertices of the graph (one per line)</li>
 * <li>--edges=(path): directory containing edges of the graph (pair of vertex ids per line in textformat)</li>
 * <li>--numIterations=(Integer): number of numIterations, default: 10</li>
 * <li>--dampingFactor=(Double): probability not to teleport to a random vertex, default: 0.85</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public class RandomWalkWithRestartJob extends RandomWalk {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RandomWalkWithRestartJob(), args);
  }

  @Override
  protected Vector createSeedVector(int numVertices) {
  	Vector dampingVector = new RandomAccessSparseVector(numVertices, 1);
    //dampingVector.set(sourceVertexIndex, 1.0);
    return dampingVector;
  }

  @Override
  protected void addSpecificOptions() {
  	//addOption("sourceVertexIndex", null, "index of source vertex", true);
  }

  @Override
  protected void evaluateSpecificOptions(Map<String, List<String>> parsedArgs) {
    //sourceVertexIndex = Integer.parseInt(getOption("sourceVertexIndex"));
  }

}