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
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;

/**
 * Random walk.
 * 
 * @author Younggue Bae
 */
abstract class RandomWalk extends AbstractJob {
	
	private static final Logger log = LoggerFactory.getLogger(RandomWalk.class);
	
	static final String SEED_VECTOR = "seedVector";
  static final String RANK_VECTOR = "rankVector";
  static final String TRANSITION_MATRIX = "transitionMatrix";
  static final String TRANSITION_PLUS_SEED_MATRIX = "transitionPlusSeedMatrix";

  static final String NUM_VERTICES_PARAM = AdjacencyMatrixJob.class.getName() + ".numVertices";
  static final String DAMPING_FACTOR_PARAM = AdjacencyMatrixJob.class.getName() + ".dampingFactor";
  static final String SEED_VECTOR_PARAM = RandomWalk.class.getName() + ".danglingVector";
  static final String DANGLING_VECTOR_PARAM = DanglingVertexJob.class.getName() + ".danglingVector";

  protected abstract void persistSeedVector(int numVertices) throws IOException;
  protected abstract Vector getSeedVector(int numVertices)throws IOException ;

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
    Path transitionMatrixPath = getTempPath(TRANSITION_MATRIX);
    Path transitionPlusSeedMatrixPath = getTempPath(TRANSITION_PLUS_SEED_MATRIX);
    Path vertexIndexPath = getTempPath(AdjacencyMatrixJob.VERTEX_INDEX);
    Path numVerticesPath = getTempPath(AdjacencyMatrixJob.NUM_VERTICES);
    Path seedVectorPath = getTempPath(SEED_VECTOR);
    Path danglingVectorPath = getTempPath(DanglingVertexJob.DANGLING_VECTOR);
    
    /* create the adjacency matrix */
    ToolRunner.run(getConf(), new AdjacencyMatrixJob(), new String[] { "--vertices", getOption("vertices"),
        "--edges", getOption("edges"), "--output", getTempPath().toString() });
    
    /* create the dangling vertices */
    ToolRunner.run(getConf(), new DanglingVertexJob(), new String[] { "--vertexIndexPath", vertexIndexPath.toString(),
      "--edges", getOption("edges"), "--output", getTempPath().toString() });

    int numVertices = HadoopUtil.readInt(numVerticesPath, getConf());
    Preconditions.checkArgument(numVertices > 0);
    
    /* persist seed vector. */
    persistSeedVector(numVertices);

    /* transpose and stochastify the adjacency matrix to create the transition matrix */
    Job createTransitionMatrix = prepareJob(adjacencyMatrixPath, transitionMatrixPath, TransposeMapper.class,
        IntWritable.class, VectorWritable.class, MergeVectorsReducer.class, IntWritable.class, VectorWritable.class);    
    // This code should be included for ClassNotFoundException!
    createTransitionMatrix.setJarByClass(RandomWalk.class);
    createTransitionMatrix.setCombinerClass(MergeVectorsCombiner.class);
    createTransitionMatrix.getConfiguration().set(NUM_VERTICES_PARAM, String.valueOf(numVertices));
    createTransitionMatrix.getConfiguration().set(DAMPING_FACTOR_PARAM, String.valueOf(dampingFactor));
    createTransitionMatrix.waitForCompletion(true);
    
    /* plus seed-dangling matrix to the transition matrix */
    Job createTransitionPlusSeedMatrix = prepareJob(transitionMatrixPath, transitionPlusSeedMatrixPath, PlusSeedMatrixMapper.class,
        IntWritable.class, VectorWritable.class, MergeVectorsReducer.class, IntWritable.class, VectorWritable.class);    
    createTransitionPlusSeedMatrix.setJarByClass(RandomWalk.class);
    createTransitionPlusSeedMatrix.setCombinerClass(MergeVectorsCombiner.class);
    createTransitionPlusSeedMatrix.getConfiguration().set(NUM_VERTICES_PARAM, String.valueOf(numVertices));
    createTransitionPlusSeedMatrix.getConfiguration().set(DAMPING_FACTOR_PARAM, String.valueOf(dampingFactor));
    createTransitionPlusSeedMatrix.getConfiguration().set(SEED_VECTOR_PARAM, seedVectorPath.toString());
    createTransitionPlusSeedMatrix.getConfiguration().set(DANGLING_VECTOR_PARAM, danglingVectorPath.toString());
    createTransitionPlusSeedMatrix.waitForCompletion(true);

    DistributedRowMatrix transitionMatrix = new DistributedRowMatrix(transitionPlusSeedMatrixPath, getTempPath(),
        numVertices, numVertices);
    transitionMatrix.setConf(getConf());

    Vector ranking = new DenseVector(numVertices).assign(1.0 / numVertices);
    Vector seedVector = getSeedVector(numVertices);

    /* power method: iterative transition-matrix times ranking-vector multiplication */
    while (numIterations-- > 0) {
    	log.debug("Iteration == " + numIterations);
      ranking = transitionMatrix.times(ranking);
      ranking = ranking.plus(seedVector.times(1 - dampingFactor));
    }

    persistVector(getConf(), getTempPath(RANK_VECTOR), ranking);

    Job vertexWithPageRank = prepareJob(vertexIndexPath, getOutputPath(), SequenceFileInputFormat.class,
        RankPerVertexMapper.class, LongWritable.class, DoubleWritable.class, TextOutputFormat.class);
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

  static class TransposeMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private int numVertices;
    private double dampingFactor;

    @SuppressWarnings("rawtypes")
		@Override
    protected void setup(Mapper.Context ctx) throws IOException, InterruptedException {
      numVertices = Integer.parseInt(ctx.getConfiguration().get(NUM_VERTICES_PARAM));
      dampingFactor = Double.parseDouble(ctx.getConfiguration().get(DAMPING_FACTOR_PARAM));
    }

    @Override
    protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
      int rowIndex = r.get();

      Vector row = v.get();
      /* divide by out-degree */
      row = row.normalize(1);	
      
      if (dampingFactor != 1.0) {
        row.assign(Functions.MULT, dampingFactor);
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
  
  static class PlusSeedMatrixMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private int numVertices;
    private double dampingFactor;
    private Vector seedVector;
    private Vector danglingVector;

    @SuppressWarnings("rawtypes")
		@Override
    protected void setup(Mapper.Context ctx) throws IOException, InterruptedException {
    	Configuration conf = ctx.getConfiguration();
    	
      numVertices = Integer.parseInt(conf.get(NUM_VERTICES_PARAM));
      dampingFactor = Double.parseDouble(ctx.getConfiguration().get(DAMPING_FACTOR_PARAM));
      Path seedVectorPath = new Path(conf.get(SEED_VECTOR_PARAM));
      Path danglingVectorPath = new Path(conf.get(DANGLING_VECTOR_PARAM));
      
      seedVector = getSeedVector(conf, seedVectorPath, numVertices);
      danglingVector = getDanglingVector(conf, danglingVectorPath, numVertices);
    }

    @Override
    protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
      int rowIndex = r.get();
      
      double seedValue = seedVector.getQuick(rowIndex);
      Vector weightRowVector = danglingVector.clone();
      weightRowVector.assign(Functions.MULT, seedValue);
      
      if (dampingFactor != 1.0) {
      	weightRowVector.assign(Functions.MULT, dampingFactor);
      }

      Vector row = v.get();
      row.plus(weightRowVector);
      
      ctx.write(r, new VectorWritable(row));
    }
    
		private Vector getSeedVector(Configuration conf, Path seedVectorPath, int numVertices) throws IOException {
			DataInputStream in = null;
			Vector values;
			try {
				in = FileSystem.get(seedVectorPath.toUri(), conf).open(seedVectorPath);
				values = VectorWritable.readVector(in);
			} finally {
				Closeables.closeQuietly(in);
			}
			return values;
		}
    
    private Vector getDanglingVector(Configuration conf, Path danglingVectorPath, int numVertices) {
    	DenseVector danglingVector = new DenseVector(numVertices);
    	
      for (Pair<IntWritable,IntWritable> indexAndDangling :
          new SequenceFileIterable<IntWritable, IntWritable>(danglingVectorPath, true, conf)) {
      	int vertexIndex = indexAndDangling.getFirst().get();
      	int dangling = indexAndDangling.getSecond().get();
      	danglingVector.setQuick(vertexIndex, dangling);
      }
      
      return danglingVector;
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
