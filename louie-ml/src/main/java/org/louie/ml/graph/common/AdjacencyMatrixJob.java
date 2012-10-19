package org.louie.ml.graph.common;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntIntHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;

/**
 * <p>Distributed computation of the adjacency matrix of a graph, see http://en.wikipedia.org/wiki/Adjacency_matrix
 *
 * <p>This job outputs {@link org.apache.hadoop.io.SequenceFile}s an {@link IntWritable} as key and a {@link VectorWritable}  as value</p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 *   <li>--output=(path): output path where the resulting matrix and the number of vertices should be written</li>
 *   <li>--vertices=(path): file containing a list of all vertices</li>
 *   <li>--edges=(path): Directory containing edges of the graph</li>
 *   <li>--symmetric = (boolean) produce a symmetric adjacency matrix (corresponds to an undirected graph)</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public class AdjacencyMatrixJob extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(AdjacencyMatrixJob.class);

  public static final String NUM_VERTICES = "numVertices.bin";
  public static final String ADJACENCY_MATRIX = "adjacencyMatrix";
  public static final String VERTEX_INDEX = "vertexIndex";
  public static final String VERTEX_VALUE = "vertexValue";
  
  static final String NUM_VERTICES_PARAM = AdjacencyMatrixJob.class.getName() + ".numVertices";
  static final String VERTEX_INDEX_PARAM = AdjacencyMatrixJob.class.getName() + ".vertexIndex";
  static final String SYMMETRIC_PARAM = AdjacencyMatrixJob.class.getName() + ".symmetric";
  
  /*
   * Whether or not to use a continuous version of the PageRank algorithm, 
   * If set to true, transition value will be set to edge weight value in adjacency matrix, 
   * otherwise, the transition value will be set to 1.
   */
  static final String CONTINUOUS = AdjacencyMatrixJob.class.getName() + ".continuous";
  /*
   * A field index with edge weight in edges input file.
   */
	static final String EDGE_WEIGHT_FIELD = AdjacencyMatrixJob.class.getName() + ".edgeWeightField";
	/*
	 * A edge weight threshold how two vertices must be to be considered "connected".
	 * This is only used in the continuous version of PageRank.
	 */
	static final String EDGE_WEIGHT_THRESHOLD = AdjacencyMatrixJob.class.getName() + ".edgeWeightThreshold";
	
  private static final Pattern SEPARATOR = Pattern.compile("[\t,]");

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new AdjacencyMatrixJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addOption("vertices", null, "a text file containing all vertices of the graph (one per line)", true);
    addOption("edges", null, "text files containing the edges of the graph (vertexA,vertexB per line)", true);
    addOption("symmetric", null, "produce a symmetric adjacency matrix (corresponds to an undirected graph)",
        String.valueOf(false));
    addOption("continuous", null, "use a continuous version of the PageRank", String.valueOf(false));
    addOption("vertexValueField", null, "a field index with value in vertices input file", String.valueOf(-1));
    addOption("edgeWeightField", null, "a field index with weight in edges input file", String.valueOf(-1));
    addOption("edgeWeightThreshold", null, "a edge weight threshold", String.valueOf(-1));

    addOutputOption();

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path vertices = new Path(getOption("vertices"));
    Path edges = new Path(getOption("edges"));
    boolean symmetric = Boolean.parseBoolean(getOption("symmetric"));
    
    log.info("Indexing vertices sequentially, this might take a while...");
    int vertexValueFieldIndex = Integer.parseInt(getOption("vertexValueField"));
    int numVertices = indexVertices(vertices, getOutputPath(VERTEX_INDEX));
    indexVerticesValues(vertices, getOutputPath(VERTEX_VALUE), vertexValueFieldIndex);

    HadoopUtil.writeInt(numVertices, getOutputPath(NUM_VERTICES), getConf());
    Preconditions.checkArgument(numVertices > 0);

    log.info("Found " + numVertices + " vertices, creating adjacency matrix...");
    
    Job createAdjacencyMatrix = prepareJob(edges, getOutputPath(ADJACENCY_MATRIX), TextInputFormat.class,
    		VectorizeEdgesMapper.class, IntWritable.class, VectorWritable.class, VectorSumReducer.class,
        IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class);
    createAdjacencyMatrix.setJarByClass(AdjacencyMatrixJob.class);
    createAdjacencyMatrix.setCombinerClass(VectorSumReducer.class);
    Configuration createAdjacencyMatrixConf = createAdjacencyMatrix.getConfiguration();
    createAdjacencyMatrixConf.set(NUM_VERTICES_PARAM, String.valueOf(numVertices));
    createAdjacencyMatrixConf.set(VERTEX_INDEX_PARAM, getOutputPath(VERTEX_INDEX).toString());
    createAdjacencyMatrixConf.setBoolean(SYMMETRIC_PARAM, symmetric);
    createAdjacencyMatrixConf.set(CONTINUOUS, getOption("continuous"));
    createAdjacencyMatrixConf.set(EDGE_WEIGHT_FIELD, getOption("edgeWeightField"));
    createAdjacencyMatrixConf.set(EDGE_WEIGHT_THRESHOLD, getOption("edgeWeightThreshold"));
    createAdjacencyMatrix.waitForCompletion(true);

    return 0;
  }

  //TODO do this in parallel?
  private int indexVertices(Path verticesPath, Path indexPath) throws IOException {
    FileSystem fs = FileSystem.get(verticesPath.toUri(), getConf());
    SequenceFile.Writer writer = null;
    int index = 0;

    try {
      writer = SequenceFile.createWriter(fs, getConf(), indexPath, IntWritable.class, IntWritable.class);

      for (FileStatus fileStatus : fs.listStatus(verticesPath)) {
        InputStream in = null;
        try {
          in = HadoopUtil.openStream(fileStatus.getPath(), getConf());
          for (String line : new FileLineIterable(in)) {
          	String[] tokens = SEPARATOR.split(line.toString());
          	int id = Integer.parseInt(tokens[0]);
            //writer.append(new IntWritable(index++), new IntWritable(Integer.parseInt(line)));
          	writer.append(new IntWritable(index++), new IntWritable(id));
          }
        } finally {
          Closeables.closeQuietly(in);
        }
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
    
    return index;
  }
  
  private void indexVerticesValues(Path verticesPath, Path valuePath, int valueFieldIndex) throws IOException {
  	if (valueFieldIndex < 0) {
  		return;
  	}
  	
    FileSystem fs = FileSystem.get(verticesPath.toUri(), getConf());
    SequenceFile.Writer writer = null;
    int index = 0;

    try {
      writer = SequenceFile.createWriter(fs, getConf(), valuePath, IntWritable.class, DoubleWritable.class);

      for (FileStatus fileStatus : fs.listStatus(verticesPath)) {
        InputStream in = null;
        try {
          in = HadoopUtil.openStream(fileStatus.getPath(), getConf());
          for (String line : new FileLineIterable(in)) {
          	String[] tokens = SEPARATOR.split(line.toString());
          	log.info("line == " + line);
          	log.info("tokens[" + valueFieldIndex + "] == " + tokens[valueFieldIndex]);
          	double value = Double.parseDouble(tokens[valueFieldIndex]);
          	writer.append(new IntWritable(index++), new DoubleWritable(value));
          }
        } finally {
          Closeables.closeQuietly(in);
        }
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }

	static class VectorizeEdgesMapper extends Mapper<LongWritable, Text, IntWritable, VectorWritable> {

    private int numVertices;
    private OpenIntIntHashMap vertexIDsToIndex;
    private boolean symmetric;

    private final IntWritable row = new IntWritable();

    private static final Pattern SEPARATOR = Pattern.compile("[\t,]");
    
    private boolean continuous;
    private int edgeWeightFieldIndex;
    private double edgeWeightThreshold;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Configuration conf = ctx.getConfiguration();
      numVertices = Integer.parseInt(conf.get(NUM_VERTICES_PARAM));
      symmetric = conf.getBoolean(SYMMETRIC_PARAM, false);
      Path vertexIndexPath = new Path(conf.get(VERTEX_INDEX_PARAM));
      vertexIDsToIndex = new OpenIntIntHashMap(numVertices);
      continuous = conf.getBoolean(CONTINUOUS, false);
      edgeWeightFieldIndex = conf.getInt(EDGE_WEIGHT_FIELD, -1);
      edgeWeightThreshold = Double.parseDouble(conf.get(EDGE_WEIGHT_THRESHOLD));

      for (Pair<IntWritable,IntWritable> indexAndVertexID :
          new SequenceFileIterable<IntWritable,IntWritable>(vertexIndexPath, true, conf)) {
        vertexIDsToIndex.put(indexAndVertexID.getSecond().get(), indexAndVertexID.getFirst().get());
      }
    }

    @SuppressWarnings({ "rawtypes", "unchecked" })
		@Override
    protected void map(LongWritable offset, Text line, Mapper.Context ctx)
        throws IOException, InterruptedException {

      String[] tokens = SEPARATOR.split(line.toString());
      int rowIndex = vertexIDsToIndex.get(Integer.parseInt(tokens[0]));
      int columnIndex = vertexIDsToIndex.get(Integer.parseInt(tokens[1]));
      
      double edgeWeight = 0.0;
      if (edgeWeightFieldIndex >= 0) {
      	edgeWeight = Double.parseDouble(tokens[edgeWeightFieldIndex]);
      }
      
      Vector partialTransitionMatrixRow = new SequentialAccessSparseVector(numVertices, 1);
      if (!continuous) {
        row.set(rowIndex);

      	if (edgeWeight > edgeWeightThreshold) {
      		partialTransitionMatrixRow.setQuick(columnIndex, 1);
      	}
        ctx.write(row, new VectorWritable(partialTransitionMatrixRow));

        if (symmetric && rowIndex != columnIndex) {
          partialTransitionMatrixRow = new SequentialAccessSparseVector(numVertices, 1);
          row.set(columnIndex);
        	if (edgeWeight > edgeWeightThreshold) {
        		 partialTransitionMatrixRow.setQuick(rowIndex, 1);
        	}
          ctx.write(row, new VectorWritable(partialTransitionMatrixRow));
        }    	
      }
      else {
      	row.set(rowIndex);
    		if (edgeWeight > edgeWeightThreshold) {
    			partialTransitionMatrixRow.setQuick(columnIndex, edgeWeight);
    		}
        ctx.write(row, new VectorWritable(partialTransitionMatrixRow));

        if (symmetric && rowIndex != columnIndex) {
          partialTransitionMatrixRow = new SequentialAccessSparseVector(numVertices, 1);
          row.set(columnIndex);
        	if (edgeWeight > edgeWeightThreshold) {
        		partialTransitionMatrixRow.setQuick(rowIndex, edgeWeight);
        	}
          ctx.write(row, new VectorWritable(partialTransitionMatrixRow));
        }
      }
    }
  }

}
