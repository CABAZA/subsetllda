package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gr.auth.csd.mlkd.atypon.journal_clustering.JournalsModel;
import gr.auth.csd.mlkd.atypon.utils.Utils;
import java.io.Serializable;
import java.util.Random;
import java.util.Set;
import gr.auth.csd.mlkd.atypon.lda.DatasetBoW;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.PriorModel;

public class JournalSubsetInferenceModel extends PriorModel implements Serializable {

    static final long serialVersionUID = -7219137807901737L;
    private final double[][] alphaPrior;
    private final int[][] possibleLabels;

    public JournalSubsetInferenceModel(DatasetBoW data, String trainedModelName, int threads, int iters, int burnin, String jm, String allMesh) {
        super(data, trainedModelName, threads, iters, burnin);
        JournalsModel jM = JournalsModel.read(jm, allMesh);
        TIntObjectHashMap<TObjectDoubleHashMap<String>> journalLabelsDistributions = jM.getJlf().getFrequencies();
        this.possibleLabels = new int[M][];
        alphaPrior = new double[M][K];
        for (int d = 0; d < M; d++) {
            String journal = data.getDocs().get(d).getJournal();

            if (journal == null || journal.isEmpty() || !jM.getJlf().getJournals().contains(journal)) {
                possibleLabels[d] = new int[K];
                for (int k = 0; k < K; k++) {
                    possibleLabels[d][k] = k + 1;
                }

                super.initAlpha();
                alphaPrior[d] = alpha;
                System.out.println(d + " " + journal);
            } //journal = journal.substring(0, journal.length()-1);
            else {
                int indexOfJournal = jM.getJlf().getJournals().indexOf(journal);
                Set<String> keySet = journalLabelsDistributions.get(indexOfJournal).keySet();
                int size = 0;
                for (String l : keySet) {
                    if (((DatasetBoW)data).getLabels().getIndex(l) >= 0) {
                        size++;
                    }
                }
                possibleLabels[d] = new int[size];

                int k = 0;
                for (String l : keySet) {
                    int index = ((DatasetBoW)data).getLabels().getIndex(l);
                    if (index >= 0) {
                        possibleLabels[d][k] = index;
                        double freq = journalLabelsDistributions.get(indexOfJournal).get(l);
                        alphaPrior[d][index - 1] = 50.0 * freq + 30.0 / K;
                        k++;
                    }
                }
                System.out.println(d + " " + possibleLabels[d].length + " " + Utils.max(alphaPrior[d]));
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
                int topic = possibleLabels[d][randomIndex] - 1;
                setZInitially(d, w, topic);
            }
        }
    }

    @Override
    public void update(int d) {
        int documentLength = data.getDocs().get(d).getWords().size();
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

    protected double computeTheta(int k, int m) {
        return nd[m].get(k) + alphaPrior[m][k];
    }

}
