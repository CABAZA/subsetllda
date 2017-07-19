package gr.auth.csd.mlkd.mlclassification.labeledlda.subspace;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import java.io.Serializable;
import java.util.Random;
import gr.auth.csd.mlkd.mlclassification.labeledlda.Dataset;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.InferenceCGSpModel;
import static gr.auth.csd.mlkd.mlclassification.labeledlda.models.Model.readPhi;
import gr.auth.csd.mlkd.utils.Utils;
import java.util.ArrayList;
import java.util.Arrays;

//todo: priors, frequency*similarity, shuffle word order, shuffle doc order, cgs vs cgs_p
public class SubsetModel extends InferenceCGSpModel implements Serializable {

    static final long serialVersionUID = -7219137807901737L;
    //private final double[][] alphaPrior;
    private final int[][] possibleLabels;

    public SubsetModel(Dataset data, String trainedModelName, int threads, int iters, int burnin, String ls) {
        super();
        this.inference = true;
        this.data = data;
        K = data.getK();
        //+ allocate memory and assign values for variables		
        M = data.getDocs().size();
        V = data.getV();
        System.out.println("K " + K);
        System.out.println("V " + V);
        System.out.println("M " + M);

        String trainedPhi = trainedModelName + ".phi";
        System.out.println(trainedPhi);
        phi = readPhi(trainedPhi);
        theta = new ArrayList<>();
        for (int d = 0; d < M; d++) {
            theta.add(new TIntDoubleHashMap());
        }
        z = new TIntIntHashMap[M];
        nd = new TIntDoubleHashMap[M];
        for (int m = 0; m < M; m++) {
            nd[m] = new TIntDoubleHashMap();
            z[m] = new TIntIntHashMap();
        }

        this.niters = iters;
        this.nburnin = burnin;
        this.threads = threads;
        this.modelName = trainedModelName;

        ArrayList<TObjectDoubleHashMap<String>> a = (ArrayList<TObjectDoubleHashMap<String>>) Utils.readObject(ls);
        this.possibleLabels = new int[M][];
        //alphaPrior = new double[M][];
        for (int d = 0; d < M; d++) {
            int[] keySet = data.getDocs().get(d).getLabels();
            int size = keySet.length;
            possibleLabels[d] = new int[size];
            //alphaPrior[d] = new double[size];
            int k = 0;
            for (int index : keySet) {
                possibleLabels[d][k] = index;
//                String label = data.getLabel(index);
//                double freq = a.get(d).get(label);          
//                alphaPrior[d][k] = 50.0 * freq + 30.0 / K;
                k++;
            }
        }
    }

    @Override
    public void initialize() {
        Random r = new Random();
        for (int d = 0; d < M; d++) {
            TIntIterator it = data.getDocs().get(d).getWords().iterator();
            while (it.hasNext()) {
                int w = it.next();
                int randomIndex = r.nextInt(possibleLabels[d].length);
                int topic = possibleLabels[d][randomIndex];
                setZInitially(d, w, topic);
            }
        }
    }

    @Override
    public void update(int m) {
        TIntIterator it = data.getDocs().get(m).getWords().iterator();
        while (it.hasNext()) {
            int w = it.next();
            int topic = z[m].get(w);
            removeZi(m, w, topic);
            int[] labels = possibleLabels[m];
            int K_m = labels.length;
            double[] p = new double[K_m];
            for (int k = 0; k < K_m; k++) {
                topic = labels[k];
                double prob = phi[topic].get(w) * (nd[m].get(topic) + alpha[topic]);//alphaPrior[m][k]);
                p[k] = (k == 0) ? prob : p[k - 1] + prob;
            }

            double u = Math.random();
            for (topic = 0; topic < K_m; topic++) {
                if (p[topic] > u * p[K_m - 1]) {
                    break;
                }
            }

            if (topic == K_m) {
                topic = K_m - 1;
            }
            topic = labels[topic];
            addZi(m, w, topic);
            z[m].put(w, topic);
        }
    }

    @Override
    public ArrayList<TIntDoubleHashMap> computeTheta(int totalSamples) {
        System.out.print("Updating parameters...");
        for (int d = 0; d < M; d++) {
            int[] labels = possibleLabels[d];
            int K_m = labels.length;
            double[] p = new double[K_m];
            TIntIterator it = data.getDocs().get(d).getWords().iterator();
            while (it.hasNext()) {
                int word = it.next();
                for (int k = 0; k < K_m; k++) {
                    int topic = labels[k];
                    //p[k] = (nd[d].get(topic) + alphaPrior[d][k]) * phi[topic].get(word);
                    p[k] = (nd[d].get(topic) + alpha[topic]) * phi[topic].get(word);
                }
                p = Utils.normalize(p, 1);

                //sum probabilities over the document
                for (int k = 0; k < K_m; k++) {
                    theta.get(d).adjustOrPutValue(labels[k], p[k], p[k]);
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
}
