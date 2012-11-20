package org.louie.ml.graph.lexrank;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * An Implementation of the LexRank algorithm described in the paper
 * "LexRank: Graph-based Centrality as Salience in Text Summarization", 
 * Erkan & Radev '04, with some of our own modifications.
 * 
 * @author Younggue Bae
 */
public class LexRanker {
	
	private double similarityThreshold;
  private double dampingFactor;
  private boolean continuous;
  
  /**
   * Constructor of LexRanker.
   * 
	 * @param similarityThreshold
	 *          how similar two items must be to be considered "connected". The
	 *          LexRank paper suggests a value of 0.1.
   * @param dampingFactor 
   * 					damping factor which is typically chosen in the interval [0.8,0.9]
	 * @param continuous
	 *          whether or not to use a continuous version of the LexRank
	 *          algorithm, If set to false, all similarity links above the
	 *          similarity threshold will be considered equal; otherwise, the
	 *          similarity scores are used. The paper authors note that
	 *          non-continuous LexRank seems to perform better.
   */
  public LexRanker(double similarityThreshold, double dampingFactor, boolean continuous) {
  	this.similarityThreshold = similarityThreshold;
  	this.dampingFactor = dampingFactor;
  	this.continuous = continuous;
  }
  
	/**
	 * Runs the LexRank algorithm over a set of data. The data must have a
	 * similarity function, and it is assumed that the similarity function is
	 * symmetric. (If this is not the case, the similarity matrix will not be
	 * computed correctly.)
	 * 
	 * @param data	the data to rank
	 * @param tolerance
	 *          power iteration will stop when the difference between iterations
	 *          is less than this. (epsilon)
	 * @param maxIteration
	 *          the maximum number of iterations for which this is allowed to run.
	 *          (Yeah, proper grammar right there)
	 */
	public <T extends Similar<T>> LexRankResults<T> rank(List<T> data, double tolerance, int maxIteration) {
		LexRankResults<T> results = new LexRankResults<T>();
		if (data.size() == 0) {
			return results;
		}
		
		int dataSize = data.size();
		Matrix matrix = new DenseMatrix(dataSize, dataSize);
    double[] degree = new double[dataSize];
    for(int i = 0; i< dataSize; i++){
        degree[i] = 1;
    }
    
    double similarity;
		for (int i = 0; i < matrix.rowSize(); i++) {
			for (int j = 0; j < matrix.columnSize(); j++) {
				similarity = data.get(i).similarity(data.get(j));
				if (similarity > similarityThreshold) {
					if (continuous) {
						matrix.set(i, j, similarity);
						degree[i] += similarity;
					}
					else {
						matrix.set(i, j, 1);
						degree[i]++;
					}
				}
				else {
					matrix.set(i, j, 0);
				}
			}
		}
		
		double val, val1;
		for (int i = 0; i < matrix.rowSize(); i++) {
			for (int j = 0; j < matrix.columnSize(); j++) {
				val = (double) dampingFactor/dataSize + dampingFactor * matrix.get(i, j) / degree[i];
				//val = (double) 1/dataSize + 1 * matrix.get(i, j) / degree[i];
				//val = 1/dataSize + 1 * matrix.get(i, j) / degree[i];
				matrix.set(i, j, val);
			}
		}
		
		// Build the neighbor graph for the results.
		for (int i = 0; i < data.size(); i++) {
			for (int j = 0; j < data.size(); j++) {
				if (matrix.get(i, j) > 0) {
					List<T> neighborList = results.neighbors.get(data.get(i));
					if (neighborList == null) {
						neighborList = new ArrayList<T>();
					}
					neighborList.add(data.get(j));
					results.neighbors.put(data.get(i), neighborList);
				}
			}
		}
		
		Vector rankings = powerMethod(matrix, dataSize, tolerance, maxIteration);
		
		// Now that we have the LexRank scores, arrange them for the results.
		List<RankPair<T>> tempList = new ArrayList<RankPair<T>>();
		for (int i = 0; i < data.size(); ++i) {
			results.scores.put(data.get(i), rankings.get(i));
			tempList.add(new RankPair<T>(data.get(i), rankings.get(i)));
		}
		Collections.sort(tempList);
		Collections.reverse(tempList);
		for (RankPair<T> pair : tempList) {
			results.rankedResults.add(pair.data);
		}
		return results;
	}
	
	/**
	 * Solves for an eigenvector of a stochastic matrix using the power iteration algorithm.
	 * 
	 * For future reference, when a paper writes "M^T", that does not mean "M
	 * raised to the power of T," even if there is a variable called "t" right
	 * there. Instead, it means "M transpose." Durrr.
	 * 
	 * @param stochasticMatrix
	 *          the matrix to get the first eigenvector of
	 * @param size
	 *          how many data we've got
	 * @param tolerance
	 *          power iteration will stop when the difference between iterations
	 *          is less than this. (epsilon)
	 * @param maxIteration
	 *          the maximum number of iterations for which this is allowed to run.
	 *          (Yeah, proper grammar right there)
	 * @return Vector
	 */
	@SuppressWarnings("unused")
	private Vector powerMethod(Matrix stochasticMatrix, int size, double tolerance,
			int maxIteration) {
		Vector p0 = new DenseVector(size);
		Vector p1 = new DenseVector(size);

		for (int i = 0; i < size; i++) {
			p0.set(i, (double) 1 / size);
		}
		Matrix mt = stochasticMatrix.transpose();
		Vector pMinus;
		p1 = mt.times(p0);
		
		pMinus = p1.minus(p0);

		int iteration = 0;
		while (iteration < maxIteration) {
			p0 = p1.clone();
			p1 = mt.times(p0);
			pMinus = p1.minus(p0);
			
			if (p1.getDistanceSquared(p0) < tolerance) {
				break;
			}
			
			iteration++;
		}
		return p1;
	}
	
	/** Internal class used for sorting data by LexRank score. */
	private static class RankPair<T> implements Comparable<RankPair<T>> {
		T data;
		double score;

		public RankPair(T d, double s) {
			data = d;
			score = s;
		}

		public int compareTo(RankPair<T> other) {
			double diff = score - other.score;
			if (diff > 0.000001) {
				return 1;
			} else if (diff < -0.000001) {
				return -1;
			} else {
				return 0;
			}
		}
	}
  
}
