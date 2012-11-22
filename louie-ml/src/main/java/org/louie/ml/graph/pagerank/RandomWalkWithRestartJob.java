package org.louie.ml.graph.pagerank;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

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

	private static final Pattern SEPARATOR = Pattern.compile("[\t]");
	private Path verticesPath;
	private int vertexValueIndex;
	
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RandomWalkWithRestartJob(), args);
  }
  
  @Override
  protected void persistSeedVector(int numVertices) throws IOException {
    FileSystem fs = FileSystem.get(verticesPath.toUri(), getConf());    
    int index = 0;
    Vector seedVector = new DenseVector(numVertices);
    
    for (FileStatus fileStatus : fs.listStatus(verticesPath)) {
      InputStream in = null;
      try {
        in = HadoopUtil.openStream(fileStatus.getPath(), getConf());
        for (String line : new FileLineIterable(in)) {
        	String[] tokens = SEPARATOR.split(line.toString());
        	double value = Double.parseDouble(tokens[vertexValueIndex]);
        	seedVector.setQuick(index, value);
        	index++;
        }
      } finally {
        Closeables.closeQuietly(in);
      }
    }
  	
  	DataOutputStream out = null;
    try {
      out = fs.create(getTempPath(SEED_VECTOR), true);
      VectorWritable.writeVector(out, seedVector);
    } finally {
      Closeables.closeQuietly(out);
    }
  }

  @Override
  protected Vector getSeedVector(int numVertices) throws IOException {
  	 DataInputStream in = null;
     Vector values;
     try {
       in = FileSystem.get(getTempPath(SEED_VECTOR).toUri(), getConf()).open(getTempPath(SEED_VECTOR));
       values = VectorWritable.readVector(in);
     } finally {
       Closeables.closeQuietly(in);
     }
     return values;
  }

  @Override
  protected void addSpecificOptions() {
  	addOption("vertexValueIndex", null, "index of value field in the vertices source file", true);
  }

  @Override
  protected void evaluateSpecificOptions(Map<String, List<String>> parsedArgs) {
    vertexValueIndex = Integer.parseInt(getOption("vertexValueIndex"));
    verticesPath = new Path(getOption("vertices"));
  }

}