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
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.common.mapreduce.MergeVectorsCombiner;
import org.apache.mahout.common.mapreduce.MergeVectorsReducer;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;

/**
 * Random walk.
 * 
 * @author Younggue Bae
 */
abstract class RandomWalk extends AbstractJob {

  static final String RANK_VECTOR = "rankVector";

  static final String NUM_VERTICES_PARAM = AdjacencyMatrixJob.class.getName() + ".numVertices";
  static final String DAMPING_FACTOR_PARAM = AdjacencyMatrixJob.class.getName() + ".dampingFactor";

  protected abstract Vector createSeedVector(int numVertices);

  protected void addSpecificOptions() {}
  protected void evaluateSpecificOptions(Map<String, List<String>> parsedArgs) {}

  @Override
  public final int run(String[] args) throws Exception {
    addOutputOption();
    addOption("vertices", null, "a text file containing all vertices of the graph (one per line)", true);
    addOption("edges", null, "edges of the graph", true);
    addOption("numIterations", "it", "number of numIterations", String.valueOf(10));
    addOption("dampingFactor", "df", "a damping factor, probability not to teleport to a random vertex(default=0.85)", String.valueOf(0.85));

    addSpecificOptions();

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    evaluateSpecificOptions(parsedArgs);

    int numIterations = Integer.parseInt(getOption("numIterations"));
    double dampingFactor = Double.parseDouble(getOption("dampingFactor"));

    Preconditions.checkArgument(numIterations > 0);
    Preconditions.checkArgument(dampingFactor > 0.0 && dampingFactor <= 1.0);

    Path adjacencyMatrixPath = getTempPath(AdjacencyMatrixJob.ADJACENCY_MATRIX);
    Path transitionMatrixPath = getTempPath("transitionMatrix");
    Path vertexIndexPath = getTempPath(AdjacencyMatrixJob.VERTEX_INDEX);
    Path numVerticesPath = getTempPath(AdjacencyMatrixJob.NUM_VERTICES);
    Path danglingVertexPath = getTempPath(DanglingVertexJob.DANGLING_VERTEX);

    /* create the adjacency matrix */
    ToolRunner.run(getConf(), new AdjacencyMatrixJob(), new String[] { "--vertices", getOption("vertices"),
        "--edges", getOption("edges"), "--output", getTempPath().toString() });
    
    /* create the dangling vertices */
    ToolRunner.run(getConf(), new DanglingVertexJob(), new String[] { "--vertexIndexPath", vertexIndexPath.toString(),
      "--edges", getOption("edges"), "--output", getTempPath().toString() });

    int numVertices = HadoopUtil.readInt(numVerticesPath, getConf());
    Preconditions.checkArgument(numVertices > 0);

    /* transpose and stochastify the adjacency matrix to create the transition matrix */
    Job createTransitionMatrix = prepareJob(adjacencyMatrixPath, transitionMatrixPath, TransposeMapper.class,
        IntWritable.class, VectorWritable.class, MergeVectorsReducer.class, IntWritable.class, VectorWritable.class);
    
    /* This code should be included for ClassNotFoundException! */
    createTransitionMatrix.setJarByClass(RandomWalk.class);
    
    createTransitionMatrix.setCombinerClass(MergeVectorsCombiner.class);
    createTransitionMatrix.getConfiguration().set(NUM_VERTICES_PARAM, String.valueOf(numVertices));
    createTransitionMatrix.getConfiguration().set(DAMPING_FACTOR_PARAM, String.valueOf(dampingFactor));
    createTransitionMatrix.waitForCompletion(true);

    DistributedRowMatrix transitionMatrix = new DistributedRowMatrix(transitionMatrixPath, getTempPath(),
        numVertices, numVertices);
    transitionMatrix.setConf(getConf());

    Vector ranking = new DenseVector(numVertices).assign(1.0 / numVertices);
    Vector seedVector = createSeedVector(numVertices);
    Vector danglingVector = createDanglingVector(danglingVertexPath, numVertices);

    /* power method: iterative transition-matrix times ranking-vector multiplication */
    while (numIterations-- > 0) {
      ranking = transitionMatrix.times(ranking)
      		.plus(seedVector.times(danglingVector).times(ranking))
      		.times(dampingFactor)
      		.plus(seedVector.times(1 - dampingFactor));
    }

    persistVector(getConf(), getTempPath(RANK_VECTOR), ranking);

    Job vertexWithPageRank = prepareJob(vertexIndexPath, getOutputPath(), SequenceFileInputFormat.class,
        RankPerVertexMapper.class, LongWritable.class, DoubleWritable.class, TextOutputFormat.class);
    vertexWithPageRank.getConfiguration().set(RankPerVertexMapper.RANK_PATH_PARAM,
        getTempPath(RANK_VECTOR).toString());
    vertexWithPageRank.waitForCompletion(true);

    return 1;
  }
  
  protected Vector createDanglingVector(Path danglingVertexPath, int numVertices) {
  	DenseVector danglingVector = new DenseVector(numVertices);
  	
    for (Pair<IntWritable,IntWritable> indexAndDangling :
        new SequenceFileIterable<IntWritable, IntWritable>(danglingVertexPath, true, getConf())) {
    	int vertexIndex = indexAndDangling.getFirst().get();
    	int dangling = indexAndDangling.getSecond().get();
    	danglingVector.setQuick(vertexIndex, dangling);
    }
    
    return danglingVector;
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

  static class TransposeMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private int numVertices;

    @SuppressWarnings("rawtypes")
		@Override
    protected void setup(Mapper.Context ctx) throws IOException, InterruptedException {
      numVertices = Integer.parseInt(ctx.getConfiguration().get(NUM_VERTICES_PARAM));
    }

    @Override
    protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
      int rowIndex = r.get();

      Vector row = v.get();
      /* divide by out-degree */
      row = row.normalize(1);	

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


  public static class RankPerVertexMapper extends Mapper<IntWritable,IntWritable,IntWritable,DoubleWritable> {

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
