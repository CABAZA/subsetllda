/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gr.auth.csd.mlkd.mlclassification.labeledlda.models;

import gnu.trove.iterator.TIntIterator;
import gr.auth.csd.mlkd.atypon.utils.Utils;
import java.util.Arrays;
import java.util.Random;
import gr.auth.csd.mlkd.lda.Dataset;
import java.util.Date;

/**
 *
 * @author Yannis Papanikolaou
 */
public class DependencyModel extends /*Prior*/ InferenceModel {

    protected int[][] z2 = null; // topic assignments for labels
    protected double[][] theta2 = null; // theta: document - topic (not label!) distributions, size M x K    
    protected double[][] nd2 = null; //count how many times a topic has been assigned to a document
    protected final double[][] phi2; //topics-labels distributionsf
    private final double[][] depAlpha;
    private final double gamma;
    int T; //number of topics
    private final double eta;
    private final double a;
    private int iteration;

    public DependencyModel(Dataset data, String trainedModelName, int threads, 
            int iters, int burnin, double[][] unlabeledPhi, double g) {
        super(data, trainedModelName, threads, iters, burnin);
                phi2 = unlabeledPhi;
        T = phi2.length;
        depAlpha = new double[M][K];
        gamma = g;
        theta2 = new double[M][T];
        nd2 = new double[M][T];
        eta = 120.0;
        a = 30 / K;
    }



    @Override
    public void initialize() {
        Random r = new Random();
        for (int d = 0; d < M; d++) {
            Arrays.fill(depAlpha[d], 300.0 / K);
            int documentLength = data.getDocs().get(d).getWords().size();
            TIntIterator it = data.getDocs().get(d).getWords().iterator();
            while (it.hasNext()) {
                int w = it.next();
                //assign randomly a topic
                setZInitially(d, w, r.nextInt(K));
            }
        }
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
        //System.out.println("Updating document's "+d+" z assignments");
        int documentLength = data.getDocs().get(d).getWords().size();
        for (int w = 0; w < documentLength; w++) {

            int word = data.getDocs().get(d).getWords().get(w);
            int topic = z[d].get(w);

            removeZi(d, w, topic);

            double probs[] = new double[K];
            for (int k = 0; k < K; k++) {
                double prob = phi[k].get(word) * (nd[d].get(k) + depAlpha[d][k]);
                probs[k] = (k == 0) ? prob : probs[k - 1] + prob;
            }

            double u = Math.random();
            for (topic = 0; topic < K; topic++) {
                if (probs[topic] > u * probs[K - 1]) {
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
    protected double computeTheta(int k, int m) {
        return nd[m].get(k) + depAlpha[m][k];
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

    @Override
    public double[][] inference() {
                initAlpha();
        initialize();
        int totalSamples = (niters-nburnin)/samplingLag;
        System.out.println("Sampling " + niters + " iterations for inference!");
        for (int i = 1; i <= niters; i++) {
            if(i%1==0) System.out.println(new Date()+" "+i);
            this.iteration = i;
            exec();
            if (i > nburnin && i % samplingLag == 0) updateParams(totalSamples);
        }
        updateParams(totalSamples);
        System.out.println("Gibbs sampling for inference completed!");
        return this.getTheta();
    }
    
    

}
