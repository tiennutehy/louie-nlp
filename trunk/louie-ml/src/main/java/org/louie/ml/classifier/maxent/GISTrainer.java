package org.louie.ml.classifier.maxent;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import org.louie.ml.classifier.model.Dataset;
import org.louie.ml.classifier.model.Event;
import org.louie.ml.classifier.model.Prior;
import org.louie.ml.classifier.model.UniformPrior;

/**
 * This class implements the algorithm of Generalized Iterative Scaling.
 * 
 * @author Younggue Bae
 */
public class GISTrainer {

	private final boolean printMessages;
	
	public GISTrainer() {
		printMessages = false;
	}
	
	public GISTrainer(boolean printMessages) {
		this.printMessages = printMessages;
	}
	
  /**
   * Train a model using the GIS algorithm.
   *
   * @param iterations  The number of GIS iterations to perform.
   * @param dataset The data set used to compress events in memory.
   * @param prior The prior distribution used to train this model.
   * @return The newly trained model, which can be used immediately or saved
   *         to disk using an opennlp.maxent.io.GISModelWriter object.
   */
  public GISModel trainModel(int iterations, Dataset dataset, Prior prior) {
  	
  	int[][] contexts = dataset.getContexts();
    int[] predicateCounts = dataset.getPredCounts();
    int[] numTimesEventsSeen = dataset.getNumTimesEventsSeen();
    int numUniqueEvents = contexts.length;
    String[] outcomeLabels = dataset.getOutcomeLabels();
    int[] outcomes = dataset.getOutcomes();
    int numOutcomes = outcomeLabels.length;
    String[] predLabels = dataset.getPredLabels();
    prior.setLabels(outcomeLabels, predLabels);
    int numPreds = predLabels.length;
    
    double correctionConstant = dataset.getCorrectionConstant();

    display("\tNumber of Events: " + numUniqueEvents + "\n");
    display("\t    Number of Outcomes: " + numOutcomes + "\n");
    display("\t  Number of Predicates: " + numPreds + "\n");
 
    // set up feature arrays
    float[][] predCount = new float[numPreds][numOutcomes];
    for (int ei = 0; ei < numUniqueEvents; ei++) {
      for (int j = 0; j < contexts[ei].length; j++) {        
         predCount[contexts[ei][j]][outcomes[ei]] += numTimesEventsSeen[ei];
      }
    }
    
    // calculate expectation probabilities by each predicate and outcome
    double[][] expectation = new double[numPreds][numOutcomes];
    for (int pi = 0; pi < numPreds; pi++) {
    	for (int oi = 0; oi < numOutcomes; oi++) {
    		expectation[pi][oi] = (double) (predCount[pi][oi] / numUniqueEvents);
    		
    		display("expectation[pred=" + pi + "][outcome=" + oi +"] = " + expectation[pi][oi] + "\n");
    	}
    }
    
    // a fake "observation" to cover features which are not detected in the data.
    double smoothingObservation = 0;
    float smoothing = 0;
    for (int ei = 0; ei < numUniqueEvents; ei++) {
    	float sum = 0;
    	float[] count = predCount[ei];
    	for (int j = 0; j < count.length; j ++) {
    		sum += count[j];
    	}
    	smoothing += (float) (correctionConstant - sum);
    }
    smoothingObservation = (double) (smoothing / numUniqueEvents);
    display("expectation[smoothingObservation] = " + smoothingObservation + "\n");

    return null;
  }
 

	
  private void display(String s) {
    if (printMessages)
      System.out.print(s);
  }
  
	public static void main(String[] args) throws Exception {
		LinkedList<Event> events = new LinkedList<Event>();
		
		String[] contexts1 = { "w1", "w2", "w3" };
		Event event1 = new Event("A", contexts1);
		
		String[] contexts2 = { "w2", "w3", "w5" };
		Event event2 = new Event("A", contexts2);
		
		String[] contexts3 = { "w1", "w2" };
		Event event3 = new Event("A", contexts3);
		
		String[] contexts4 = { "w4", "w5" };
		Event event4 = new Event("B", contexts4);
		
		String[] contexts5 = { "w2", "w4", "w6" };
		Event event5 = new Event("B", contexts5);
		
		String[] contexts6 = { "w5", "w6" };
		Event event6 = new Event("B", contexts6);
		
		String[] contexts7 = { "w5", "w6" };
		Event event7 = new Event("B", contexts7);
		
		events.add(event1);
		events.add(event2);
		events.add(event3);
		events.add(event4);
		events.add(event5);
		events.add(event6);
		events.add(event7);
		
		Map<String, Integer> predicateIndex = new HashMap<String, Integer>();
		predicateIndex.put("w1", 0);
		predicateIndex.put("w2", 1);
		predicateIndex.put("w3", 2);
		predicateIndex.put("w4", 3);
		predicateIndex.put("w5", 4);
		predicateIndex.put("w6", 5);
		
		Dataset dataset = new Dataset();
		dataset.setEvents(events, predicateIndex, true);
		
		System.out.println("The labels of outcomes == " + Arrays.asList(dataset.getOutcomeLabels()));
		System.out.println("The labels of context predicates == " + Arrays.asList(dataset.getPredLabels()));
		
		GISTrainer trainer = new GISTrainer(true);
		trainer.trainModel(15, dataset, new UniformPrior());
	}
	
}
