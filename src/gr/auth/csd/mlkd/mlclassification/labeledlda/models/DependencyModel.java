/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gr.auth.csd.mlkd.mlclassification.labeledlda.models;

import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.mlclassification.labeledlda.Dataset;
import gr.auth.csd.mlkd.mlclassification.labeledlda.DatasetTfIdf;
import gr.auth.csd.mlkd.utils.Utils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Date;

/**
 *
 * @author Yannis Papanikolaou
 */
public class DependencyModel extends /*Prior*/ InferenceCGSpModel {

    protected int[][] z2 = null; // topic assignments for labels
    protected double[][] theta2 = null; // theta: document - topic (not label!) distributions, size M x K    
    protected double[][] nd2 = null; //count how many times a topic has been assigned to a document
    protected final double[][] phi2; //topics-labels distributionsf
    private double[][] depAlpha;
    private final double gamma;
    int T; //number of topics
    private final double eta;
    private final double a;
    private int iteration;

    public DependencyModel(Dataset data, int niters,
            int nburnin, String modelName, double[][] unlabeledPhi, double g, int threads) {
        super(data, modelName, threads, niters, nburnin);
        phi2 = unlabeledPhi;
        T = phi2.length;
        gamma = g;
        theta2 = new double[M][T];
        nd2 = new double[M][T];
        eta = 120.0;
        this.a = 30 / K;
    }   

    protected void initialize2() {
        Random r = new Random();
        z2 = new int[M][];
        for (int d = 0; d < M; d++) {
            int documentLength = data.getDocs().get(d).getWords().size();
            z2[d] = new int[documentLength];
//            iterate over the assigned labels and assign randomly to them a topic
            for (int w = 0; w < documentLength; w++) {
                int t = r.nextInt(T);
                z2[d][w] = t;
                nd2[d][t]++;
            }
        }
        super.initAlpha();
    }

    @Override
    public void update(int d) {
        double[] p = new double[K];
        TIntIterator it = data.getDocs().get(d).getWords().iterator();
        while (it.hasNext()) {
            int w = it.next();
            int topic = z[d].get(w);
            removeZi(d, w, topic);
            for (int k = 0; k < K; k++) {
                double prob = phi[k].get(w) * (nd[d].get(k) + depAlpha[d][k]);
                p[k] = (k == 0) ? prob : p[k - 1] + prob;
            }

            double u = Math.random();
            for (topic = 0; topic < K; topic++) {
                if (p[topic] > u * p[K - 1]) {
                    break;
                }
            }
            if (topic == K) {
                topic = K - 1;
            }

            z[d].put(w, topic);
            addZi(d, w, topic);
        }
        if (this.iteration == 3 && (d == M - 1)) {
            initialize2();
        }
        if (this.iteration > 3 && (d == M - 1)) {
            update2(d);
            initAlpha();
        }
    }
    
    public void update2(int d) {
        //System.out.println("Updating document's "+d+" z' assignments");
        int documentLength = data.getDocs().get(d).getWords().size();
        for (int w = 0; w < documentLength; w++) {
            int k = z[d].get(w);
            int t = z2[d][w];
            nd2[d][t]--;

            double probs[] = new double[T];
            for (t = 0; t < T; t++) {
                double prob = phi2[t][k] * (nd2[d][t] + gamma);
                probs[t] = (t == 0) ? prob : probs[t - 1] + prob;
            }

            double u = Math.random();
            for (t = 0; t < T; t++) {
                if (probs[t] > u * probs[T - 1]) {
                    break;
                }
            }
            if (t == T) {
                t = T - 1;
            }
            z2[d][w] = t;
            nd2[d][t]++;
        }
    }

    @Override

    protected ArrayList<TIntDoubleHashMap> computeTheta(int totalSamples) {
        System.out.print("Updating parameters...");
        for (int d = 0; d < M; d++) {
            double tempTheta[] = new double[K];
            double[] p = new double[K];
            TIntArrayList words = data.getDocs().get(d).getWords();
            for (int w = 0; w < words.size(); w++) {
                int word = data.getDocs().get(d).getWords().get(w);
                int topic = z[d].get(w);
                nd[d].adjustValue(topic, -1);
                for (int k = 0; k < K; k++) {
                    p[k] = (nd[d].get(k) + depAlpha[d][k]) * phi[k].get(word);
                }
                nd[d].adjustValue(topic, 1);
                //p = Utils.normalize(p, 1);

                //sum probabilities over the document
                for (int k = 0; k < K; k++) {
                    theta.get(d).adjustOrPutValue(k + 1, p[k], p[k]);
                }
            }
        }
        if (numSamples == totalSamples) {
            for (int m = 0; m < M; m++) {
                theta.set(m, Utils.normalize(theta.get(m), 1.0));
            }
        }
        return theta;
    }

    protected double[][] computeTheta2(boolean average, boolean finalSample) {
        double[] tempTheta = new double[T];
        for (int d = 0; d < M; d++) {
            for (int t = 0; t < T; t++) {
                tempTheta[t] = nd2[d][t] + gamma;
            }
            tempTheta = Utils.normalize(tempTheta, 1);
            //System.out.println(d+" "+Arrays.toString(tempTheta));
            for (int t = 0; t < T; t++) {
                if (average) {
                    theta2[d][t] += tempTheta[t];
                } else {
                    theta2[d][t] = tempTheta[t];
                }
            }
        }
        //System.out.println(Arrays.toString(theta[0]));
        if (finalSample) {
            for (int d = 0; d < M; d++) {
                theta2[d] = Utils.normalize(theta2[d], 1);
            }
        }
        return theta2;
    }

   @Override
    public void initAlpha() {
        System.out.println("Calculating alpha...");
        depAlpha = new double[M][K];
//        double sumOfFrequencies = 0;
//        TIntDoubleHashMap labels = ((DatasetTfIdf)data).getLabels();
//        TIntDoubleIterator it = labels.iterator();
//        while(it.hasNext()) {
//            it.advance();
//            sumOfFrequencies += it.value();
//        }
//        for (int d = 0; d < M; d++) {
//        it = labels.iterator();
//        while(it.hasNext()) {
//            it.advance();
//            int k = it.key();
//            double frequency = it.value();
//            depAlpha[d][k-1] = 50.0 * frequency / sumOfFrequencies + 30.0 / K;
//        }
//        }
        
        theta2 = computeTheta2(false, true);
        double[][] innerProduct = new double[M][K];
        for (int d = 0; d < M; d++) {
            double sum = 0;
            for (int k = 0; k < K; k++) {
                double ip = 0;
                for (int t = 0; t < T; t++) {
                    ip += theta2[d][t] * phi2[t][k];
                }
                innerProduct[d][k] = ip;
                sum += innerProduct[d][k];
            }

            for (int k = 0; k < K; k++) {
                depAlpha[d][k] = eta * (innerProduct[d][k] / sum) + a;
            }
            // System.out.println(d+" "+" "+Utils.maxIndex(depAlpha[d])+" "+(depAlpha[d][Utils.maxIndex(depAlpha[d])])+" "+data.getLabel(Utils.maxIndex(depAlpha[d]) ));
        }

    }
}
