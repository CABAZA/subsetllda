package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gr.auth.csd.mlkd.atypon.utils.Utils;
import java.io.Serializable;
import java.util.Random;
import gr.auth.csd.mlkd.atypon.lda.Dataset;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.OnlyCGS_pPriorModel;
import java.util.ArrayList;

public class SubsetModel extends OnlyCGS_pPriorModel implements Serializable {

    static final long serialVersionUID = -7219137807901737L;
    private final double[][] alphaPrior;
    private final int[][] possibleLabels;

    public SubsetModel(Dataset data, String trainedModelName, int threads, int iters, int burnin, String ls) {
        super(data, trainedModelName, threads, iters, burnin);
        ArrayList<TObjectDoubleHashMap<String>> a = (ArrayList<TObjectDoubleHashMap<String>>) Utils.readObject(ls);
        this.possibleLabels = new int[M][];
        alphaPrior = new double[M][K];
        for (int d = 0; d < M; d++) {

            int[] keySet = data.getDocs().get(d).getLabels();
            int size = keySet.length;
            possibleLabels[d] = new int[size];

            int k = 0;
            for (int index : keySet) {
                possibleLabels[d][k] = index;
                String label = data.getLabel(index);
                double freq = a.get(d).get(label);
                alphaPrior[d][index - 1] = 50.0*freq + 30.0 / K;
                k++;
            }
            //System.out.println(d + " " + possibleLabels[d].length + " " + 
            //        Utils.max(alphaPrior[d])+" "+Arrays.toString(alphaPrior[d]));

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
                int topic = possibleLabels[d][randomIndex] - 1;
                setZInitially(d, w, topic);
            }
        }
    }

    @Override
    public void update(int d) {
        int documentLength = data.getDocs().get(d).getWords().size();
        //data.getDocs().get(d).getWords().shuffle(new Random());
        for (int w = 0; w < documentLength; w++) {
            int word = data.getDocs().get(d).getWords().get(w);

            int topic = z[d].get(w);
            removeZi(d, w, topic);

            int[] labels = possibleLabels[d];
            int K_m = labels.length;

            double probs[] = new double[K_m];
            for (int k = 0; k < K_m; k++) {
                topic = labels[k] - 1;
                double prob = phi[topic].get(word) * (nd[d].get(topic) + alphaPrior[d][topic]);
                //double prob = phi[topic].get(word) * (nd[d][topic] + alpha[topic]);
                probs[k] = (k == 0) ? prob : probs[k - 1] + prob;
            }

            double u = Math.random();
            for (topic = 0; topic < K_m; topic++) {
                if (probs[topic] > u * probs[K_m - 1]) {
                    break;
                }
            }
            if (topic == K_m) {
                topic = K_m - 1;
            }
            topic = labels[topic] - 1;
            z[d].put(w, topic);
            addZi(d, w, topic);
        }
    }

    @Override
    public void updateParams(int totalSamples) {
        System.out.print("Updating parameters...");
        for (int d = 0; d < M; d++) {
            int[] labels = possibleLabels[d];
            int K_m = labels.length;
            double[] p = new double[K_m];
            TIntIterator it = data.getDocs().get(d).getWords().iterator();
            while(it.hasNext()) {
                int word = it.next();
//                int t = z[d].get(word);
//                nd[d].adjustValue(t, -1);
                for (int k = 0; k < K_m; k++) {
                    int topic = labels[k] - 1;
                    p[k] = (nd[d].get(topic) + alphaPrior[d][topic]) * phi[topic].get(word);
                }
                //nd[d].adjustValue(t, 1);
                p = Utils.normalize(p, 1);

                //sum probabilities over the document
                for (int k = 0; k < K_m; k++) {
                    int topic = labels[k] - 1;
                    theta[d][topic] += p[k];
                }
            }
        }

        //System.out.println(Arrays.toString(theta[0]));
        if (numSamples == totalSamples) {
            for (int m = 0; m < M; m++) {
                theta[m] = Utils.normalize(theta[m], 1.0);
            }
        }

        //System.out.println(Arrays.toString(theta[0]));
        //return theta;
    }
}
