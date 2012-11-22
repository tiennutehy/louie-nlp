package org.louie.ml.graph.pagerank;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntIntHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

/**
 * <p>Creating the dangling vertices.
 * <p>If a vertex has out-degree "1", otherwise "0".</p>
 * 
 * <ol>
 *   <li>--output=(path): output path</li>
 *   <li>--vertices=(path): file containing a list of all vertices</li>
 *   <li>--edges=(path): Directory containing edges of the graph</li>
 * </ol>
 * @author Younggue Bae
 */
public class DanglingVertexJob extends AbstractJob {

	private static final Logger log = LoggerFactory.getLogger(DanglingVertexJob.class);

  public static final String DANGLING_VECTOR = "danglingVector";
  
  private static final Pattern SEPARATOR = Pattern.compile("[\t]");

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new DanglingVertexJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addOption("vertexIndexPath", null, "a vertex index file", true);
    addOption("edges", null, "text files containing the edges of the graph (vertexA,vertexB per line)", true);

    addOutputOption();

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path verticesIndexPath = new Path(getOption("vertexIndexPath"));
    Path edgesPath = new Path(getOption("edges"));
    
    log.info("Creating dangling vertices...");
    
    OpenIntIntHashMap vertexIDsToIndex = this.loadVerticesIndex(verticesIndexPath);
    this.persistDanglingVertices(vertexIDsToIndex, edgesPath, getOutputPath(DANGLING_VECTOR));
    
    log.info("Created dangling vertices.");

    return 0;
  }
  
  private OpenIntIntHashMap loadVerticesIndex(Path verticesIndexPath) {
  	OpenIntIntHashMap vertexIDsToIndex = new OpenIntIntHashMap();
    for (Pair<IntWritable,IntWritable> indexAndVertexID :
        new SequenceFileIterable<IntWritable, IntWritable>(verticesIndexPath, true, getConf())) {
      vertexIDsToIndex.put(indexAndVertexID.getSecond().get(), indexAndVertexID.getFirst().get());
    }
    
    return vertexIDsToIndex;
  }

  private void persistDanglingVertices(OpenIntIntHashMap vertexIDsToIndex, Path edgesPath, Path danglingPath) throws IOException {
  	OpenIntIntHashMap vertexIndexToDangling = new OpenIntIntHashMap();
  	FileSystem fs = FileSystem.get(edgesPath.toUri(), getConf());
    SequenceFile.Writer writer = null;

    try {
      writer = SequenceFile.createWriter(fs, getConf(), danglingPath, IntWritable.class, IntWritable.class);

      for (FileStatus fileStatus : fs.listStatus(edgesPath)) {
        InputStream in = null;
        try {
          in = HadoopUtil.openStream(fileStatus.getPath(), getConf());
          for (String line : new FileLineIterable(in)) {
          	String[] tokens = SEPARATOR.split(line.toString());
          	int fromIndex = vertexIDsToIndex.get(Integer.parseInt(tokens[0]));
            @SuppressWarnings("unused")
						int toIndex = vertexIDsToIndex.get(Integer.parseInt(tokens[1]));
            
            vertexIndexToDangling.put(fromIndex, 1);
          }
        } finally {
          Closeables.closeQuietly(in);
        }
      }
      IntArrayList keys = vertexIDsToIndex.keys();
      for (int i = 0; i < keys.size(); i++) {
      	int vertexId = keys.get(i);
      	int vertexIndex = vertexIDsToIndex.get(vertexId);
      	int dangling = vertexIndexToDangling.get(vertexIndex);
      	
      	if (dangling == 1) {
      		writer.append(new IntWritable(vertexIndex), new IntWritable(1));
      	}
      	else {
      		writer.append(new IntWritable(vertexIndex), new IntWritable(0));
      	}
      }     
    } finally {
      Closeables.closeQuietly(writer);
    }
  }

}
