package org.louie.ml.classifier.model;

/**
 * Class which associates a real valued parameter or expected value with a particular contextual
 * predicate or feature. This is used to store maxent model parameters as well as model and empirical
 * expected values.
 * 
 * @author Younggue Bae
 */
public class Context {

  /** The real valued parameters or expected values for this context. */
  protected double[] parameters;
  /** The outcomes which occur with this context. */
  protected int[] outcomes;
  
  /**
   * Creates a new parameters object with the specified parameters associated with the specified
   * outcome pattern.
   * @param outcomePattern Array of outcomes for which parameters exists for this context.
   * @param parameters Parameters for the outcomes specified.
   */
  public Context(int[] outcomePattern, double[] parameters) {
    this.outcomes = outcomePattern;
    this.parameters = parameters;
  }
  
  /**
   * Returns the outcomes for which parameters exists for this context.
   * @return Array of outcomes for which parameters exists for this context.
   */
  public int[] getOutcomes() {
    return outcomes;
  }
  
  /**
   * Returns the parameters or expected values for the outcomes which occur with this context.
   * @return Array of parameters for the outcomes of this context.
   */
  public double[] getParameters() {
    return parameters;
  }
}
