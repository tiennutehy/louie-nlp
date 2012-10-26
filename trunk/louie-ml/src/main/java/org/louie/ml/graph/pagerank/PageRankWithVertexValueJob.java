package org.louie.ml.graph.pagerank;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.louie.ml.graph.common.AdjacencyMatrixJob;

import com.google.common.io.Closeables;

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
 * <li>--vertexValueFieldIndex=(int): index of the vertex value field for the random walk</li>
 * <li>--edges=(path): directory containing edges of the graph (pair of vertex ids per line in textformat)</li>
 * <li>--numIterations=(Integer): number of numIterations, default: 10</li>
 * <li>--stayingProbability=(Double): probability not to teleport to a random vertex, default: 0.85</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public class PageRankWithVertexValueJob extends RandomWalk {
	
	private double vertexValueNormalizer;
	
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new PageRankWithVertexValueJob(), args);
  }

  @Override
  protected Vector createDampingVector(int numVertices, double stayingProbability) {
    Vector dampingVector = new DenseVector(numVertices).assign((1.0 - stayingProbability) / numVertices);

    return dampingVector;
  }
  
  @Override
  protected Vector createVertexValueVector(int numVertices) {
    Vector vertexValueVector = new DenseVector(numVertices).assign(0);
    try {
    	Vector verticesValuesVector = loadVertexValueVector(getTempPath(AdjacencyMatrixJob.VERTEX_VALUE));
    	vertexValueVector = vertexValueVector.plus(verticesValuesVector.times(vertexValueNormalizer / numVertices));
    } catch (IOException e) {
    	System.err.println(e.getMessage());
    	e.printStackTrace();
    }
    return vertexValueVector;
  }

  @Override
  protected void addSpecificOptions() {
    addOption("vertexValueField", null, "index of the vertex value field", true);
    addOption("vertexValueNormalizer", null, "vertex value normalizer", true);
  }

  @Override
  @SuppressWarnings("unused")
  protected void evaluateSpecificOptions(Map<String, List<String>> parsedArgs) {
		int vertexValueFieldIndex = Integer.parseInt(getOption("vertexValueField"));
		vertexValueNormalizer = Double.parseDouble(getOption("vertexValueNormalizer"));
  }
  
  //getOutputPath(VERTEX_INDEX)
  
  private Vector loadVertexValueVector(Path vertexValuePath) throws IOException {
    DataInputStream in = null;
    Vector values;
    try {
      in = FileSystem.get(vertexValuePath.toUri(), getConf()).open(vertexValuePath);
      values = VectorWritable.readVector(in);
    } finally {
      Closeables.closeQuietly(in);
    }
    return values;
  }

}