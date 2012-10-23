package org.louie.ml.graph.pagerank;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.mapreduce.MergeVectorsCombiner;
import org.apache.mahout.common.mapreduce.MergeVectorsReducer;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.louie.ml.graph.common.AdjacencyMatrixJob;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;

abstract class RandomWalk extends AbstractJob {

  static final String RANK_VECTOR = "rankVector";

  static final String NUM_VERTICES_PARAM = AdjacencyMatrixJob.class.getName() + ".numVertices";
  
  /*
   * A damping factor which is typically chosen in the interval [0.8,0.9].
   * Probability not to teleport to a random vertex.
   */
  static final String STAYING_PROBABILITY_PARAM = AdjacencyMatrixJob.class.getName() + ".stayingProbability";

  protected abstract Vector createDampingVector(int numVertices, double stayingProbability);
  
  protected abstract Vector createVertexValueVector(int numVertices);

  protected void addSpecificOptions() {}
  protected void evaluateSpecificOptions(Map<String, List<String>> parsedArgs) {}

  @Override
  public final int run(String[] args) throws Exception {
    addOutputOption();
    addOption("vertices", null, "a text file containing all vertices of the graph (one per line)", true);
    addOption("edges", null, "edges of the graph", true);
    addOption("numIterations", "it", "number of numIterations", String.valueOf(10));
    addOption("stayingProbability", "tp", "probability not to teleport to a random vertex", String.valueOf(0.85));
    addOption("continuous", null, "use a continuous version of the PageRank", String.valueOf(false));
    addOption("edgeWeightField", null, "a field index with weight in edges input file", String.valueOf(-1));
    addOption("edgeWeightThreshold", null, "a edge weight threshold", String.valueOf(-1));

    addSpecificOptions();

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    evaluateSpecificOptions(parsedArgs);

    int numIterations = Integer.parseInt(getOption("numIterations"));
    double stayingProbability = Double.parseDouble(getOption("stayingProbability"));

    Preconditions.checkArgument(numIterations > 0);
    Preconditions.checkArgument(stayingProbability > 0.0 && stayingProbability <= 1.0);

    Path adjacencyMatrixPath = getTempPath(AdjacencyMatrixJob.ADJACENCY_MATRIX);
    Path transitionMatrixPath = getTempPath("transitionMatrix");
    Path vertexIndexPath = getTempPath(AdjacencyMatrixJob.VERTEX_INDEX);
    Path numVerticesPath = getTempPath(AdjacencyMatrixJob.NUM_VERTICES);
    
    /* create the adjacency matrix */
    if (getOption("vertexValueField") == null) {
      ToolRunner.run(getConf(), new AdjacencyMatrixJob(), new String[] { "--vertices", getOption("vertices"),
        "--edges", getOption("edges"), "--output", getTempPath().toString(), 
        "--continuous", getOption("continuous"), 
        "--edgeWeightField", getOption("edgeWeightField"), 
        "--edgeWeightThreshold", getOption("edgeWeightThreshold") });   	
    }
    else {
	    ToolRunner.run(getConf(), new AdjacencyMatrixJob(), new String[] { "--vertices", getOption("vertices"),
	        "--edges", getOption("edges"), "--output", getTempPath().toString(), 
	        "--continuous", getOption("continuous"), 
	        "--vertexValueField", getOption("vertexValueField"), 
	        "--edgeWeightField", getOption("edgeWeightField"), 
	        "--edgeWeightThreshold", getOption("edgeWeightThreshold") });
    }

    int numVertices = HadoopUtil.readInt(numVerticesPath, getConf());
    Preconditions.checkArgument(numVertices > 0);

    /* transpose and stochastify the adjacency matrix to create the transition matrix */
    Job createTransitionMatrix = prepareJob(adjacencyMatrixPath, transitionMatrixPath, TransposeMapper.class,
        IntWritable.class, VectorWritable.class, MergeVectorsReducer.class, IntWritable.class, VectorWritable.class);
    createTransitionMatrix.setJarByClass(RandomWalk.class);
    createTransitionMatrix.setCombinerClass(MergeVectorsCombiner.class);
    createTransitionMatrix.getConfiguration().set(NUM_VERTICES_PARAM, String.valueOf(numVertices));
    createTransitionMatrix.getConfiguration().set(STAYING_PROBABILITY_PARAM, String.valueOf(stayingProbability));
    createTransitionMatrix.waitForCompletion(true);

    DistributedRowMatrix transitionMatrix = new DistributedRowMatrix(transitionMatrixPath, getTempPath(),
        numVertices, numVertices);
    transitionMatrix.setConf(getConf());

    Vector ranking = new DenseVector(numVertices).assign(1.0 / numVertices);
    Vector dampingVector = createDampingVector(numVertices, stayingProbability);
    Vector vertexValueVector = createVertexValueVector(numVertices);
    
    /*
    Vector amplifiedVertexValueVector = vertexValueVector.clone();
    for (int i = 0; i < amplifiedVertexValueVector.size(); i++) {
    	if (amplifiedVertexValueVector.get(i) > 0.0) {
    		amplifiedVertexValueVector.setQuick(i, (amplifiedVertexValueVector.get(i) + 1));
    	}
    }
    */
    
    /* power method: iterative transition-matrix times ranking-vector multiplication */
    while (numIterations-- > 0) {
    	ranking = transitionMatrix.times(ranking).plus(dampingVector);
     	if (vertexValueVector != null) {
     		//ranking = ranking.plus(amplifiedVertexValueVector);
     		ranking = ranking.times(0.3).plus(vertexValueVector.times(0.7));
    	}
    }

    persistVector(getConf(), getTempPath(RANK_VECTOR), ranking);

    Job vertexWithPageRank = prepareJob(vertexIndexPath, getOutputPath(), SequenceFileInputFormat.class,
        RankPerVertexMapper.class, LongWritable.class, DoubleWritable.class, TextOutputFormat.class);
    vertexWithPageRank.setJarByClass(RandomWalk.class);
    vertexWithPageRank.getConfiguration().set(RankPerVertexMapper.RANK_PATH_PARAM,
        getTempPath(RANK_VECTOR).toString());
    vertexWithPageRank.waitForCompletion(true);

    return 1;
  }

  static void persistVector(Configuration conf, Path path, Vector vector) throws IOException {
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    DataOutputStream out = null;
    try {
      out = fs.create(path, true);
      VectorWritable.writeVector(out, vector);
    } finally {
      Closeables.closeQuietly(out);
    }
  }

  static class TransposeMapper extends Mapper<IntWritable, VectorWritable, IntWritable,  VectorWritable> {

    private int numVertices;
    private double stayingProbability;

    @SuppressWarnings("rawtypes")
		@Override
    protected void setup(Mapper.Context ctx) throws IOException, InterruptedException {
      stayingProbability = Double.parseDouble(ctx.getConfiguration().get(STAYING_PROBABILITY_PARAM));
      numVertices = Integer.parseInt(ctx.getConfiguration().get(NUM_VERTICES_PARAM));
    }

    @Override
    protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
      int rowIndex = r.get();

      Vector row = v.get();
      row = row.normalize(1);
      if (stayingProbability != 1.0) {
        row.assign(Functions.MULT, stayingProbability);
      }

      Iterator<Vector.Element> it = row.iterateNonZero();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        RandomAccessSparseVector tmp = new RandomAccessSparseVector(numVertices, 1);
        tmp.setQuick(rowIndex, e.get());
        r.set(e.index());
        ctx.write(r, new VectorWritable(tmp));
      }
    }
  }


  public static class RankPerVertexMapper extends Mapper<IntWritable, IntWritable, IntWritable, DoubleWritable> {

    static final String RANK_PATH_PARAM = RankPerVertexMapper.class.getName() + ".pageRankPath";

    private Vector ranks;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Path pageRankPath = new Path(ctx.getConfiguration().get(RANK_PATH_PARAM));
      DataInputStream in = null;
      try {
        in = FileSystem.get(pageRankPath.toUri(), ctx.getConfiguration()).open(pageRankPath);
        ranks = VectorWritable.readVector(in);
      } finally {
        Closeables.closeQuietly(in);
      }
    }

    @SuppressWarnings({ "rawtypes", "unchecked" })
		@Override
    protected void map(IntWritable index, IntWritable vertex, Mapper.Context ctx)
        throws IOException, InterruptedException {
      ctx.write(vertex, new DoubleWritable(ranks.get(index.get())));
    }
  }

}
