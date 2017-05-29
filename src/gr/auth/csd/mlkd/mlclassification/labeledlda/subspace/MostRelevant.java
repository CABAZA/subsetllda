/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gr.auth.csd.mlkd.mlclassification.labeledlda.subspace;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gr.auth.csd.mlkd.utils.Timer;
import gr.auth.csd.mlkd.utils.Utils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
public abstract class MostRelevant {
    
    protected final int n;
    final String testFile;
    final String trainFile;
    protected ArrayList<TObjectDoubleHashMap<String>> labels;
    ArrayList<TIntDoubleHashMap> trainVectors;
    ArrayList<TIntDoubleHashMap> testVectors;
    TIntObjectHashMap<Set<String>> trainingFileLabels;

    public MostRelevant(int n, String testFile, String trainFile) {
        this.n = n;
        this.testFile = testFile;
        this.trainFile = trainFile;
    }

    public ArrayList<TIntDoubleHashMap> getTrainVectors() {
        return trainVectors;
    }

    protected ArrayList<TObjectDoubleHashMap<String>> findMostRelevant() {
        labels = new ArrayList<>(testVectors.size());
        List<RelevantCallable<Integer>> tasks = new ArrayList<>();
        for (int i = 0; i < testVectors.size(); i++) {
            RelevantCallable<Integer> c = new RelevantCallable<>(i, this);
            tasks.add(c);
        }
        ExecutorService exec = Executors.newFixedThreadPool(40);
        try {
            List<Future<TObjectDoubleHashMap<String>>> furs = exec.invokeAll(tasks);
            exec.shutdown();
            exec.awaitTermination(Integer.MAX_VALUE, TimeUnit.SECONDS);
            for (Future<TObjectDoubleHashMap<String>> f : furs) {
                labels.add(f.get());
            }
        } catch (InterruptedException | ExecutionException ex) {
            Logger.getLogger(MostRelevant.class.getName()).log(Level.SEVERE, null, ex);
        }
        //cleanRareLabels(labels, 1);
        writeFile(labels);
        //System.out.println(labels.get(0));
        return labels;
    }

    protected abstract void writeFile(ArrayList<TObjectDoubleHashMap<String>> relevantLabels);

    public ArrayList<TObjectDoubleHashMap<String>> getLabels() {
        return labels;
    }

    protected TObjectDoubleHashMap<String> processPerTestInstance(int i, double[] similarities) {
        if(i%100==0) System.out.println(new Date()+" "+i);
        TObjectDoubleHashMap<String> l = new TObjectDoubleHashMap<>();
        for (int t = 0; t < trainVectors.size(); t++) {
            similarities[t] = similarity(testVectors.get(i), trainVectors.get(t));
        }
        //System.out.println("a:" + trainingTime.durationMiliSeconds());
        //trainingTime = new Timer();
        for (int iter = 0; iter < n; iter++) {
            int maxRelevance = Utils.maxIndex(similarities);
            double sim = similarities[maxRelevance];
            similarities[maxRelevance] = Integer.MIN_VALUE;
            Set<String> set = trainingFileLabels.get(maxRelevance);
            //if(i==0) System.out.println(maxRelevance+" "+sim+" "+set.toString());
            for (String label : set) {
                l.adjustOrPutValue(label, 1, 1);
                //l.adjustOrPutValue(label, sim, sim);
            }
        }
        return l;
    }

    public double similarity(TIntDoubleHashMap get, TIntDoubleHashMap get0) {
        return Utils.cosineSimilarity(get, get0);
    }

    public abstract ArrayList<TObjectDoubleHashMap<String>> mostRelevant();
    
}
