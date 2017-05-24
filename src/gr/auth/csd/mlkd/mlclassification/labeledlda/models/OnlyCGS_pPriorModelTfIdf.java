package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models;

import gnu.trove.list.array.TIntArrayList;
import gr.auth.csd.mlkd.atypon.lda.Dataset;
import gr.auth.csd.mlkd.atypon.utils.Utils;


/**
 *
 * @author Yannis Papanikolaou
 */
public class OnlyCGS_pPriorModelTfIdf extends PriorModelTfIdf {

    public OnlyCGS_pPriorModelTfIdf() {
        super();
    }
    
    public OnlyCGS_pPriorModelTfIdf(Dataset data, String trainedModelName, int threads, int iters, int burnin) {
        super(data, trainedModelName, threads, iters, burnin);
    }
       
    @Override
    protected double[][] computeTheta(int totalSamples) {
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
                    p[k] = (nd[d].get(k) + alpha[k]) * phi[k].get(word);
                }
                nd[d].adjustValue(topic, 1);
                //p = Utils.normalize(p, 1);

                //sum probabilities over the document
                for (int k = 0; k < K; k++) {
                    theta[d][k] += p[k];
                }
            }
        }

        //System.out.println(Arrays.toString(theta[0]));
        if (numSamples == totalSamples) {
            for (int m = 0; m < M; m++) {
                for (int k = 0; k < K; k++) {
                    //average over samples
                    theta[m][k] /= numSamples;
                }
                //average over Nd and Sum_alpha
                theta[m] = Utils.normalize(theta[m], 1.0);
            }
        }

        //System.out.println(Arrays.toString(theta[0]));
        return theta;
    }
}
