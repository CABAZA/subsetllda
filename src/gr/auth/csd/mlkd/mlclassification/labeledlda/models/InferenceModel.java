package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models;

import gr.auth.csd.mlkd.atypon.lda.Dataset;
import gr.auth.csd.mlkd.atypon.utils.Utils;

/**
 *
 * @author Yannis Papanikolaou
 */
public class InferenceModel extends Model {

    public InferenceModel(Dataset data, String trainedModelName, int threads, int iters, int burnin) {
        super(data, 1, 0.01, true, trainedModelName, threads, iters, burnin);
    }

    @Override
    public void updateParams(int totalSamples) {
        System.out.print("Updating parameters...");
        numSamples++;
        for (int m = 0; m < M; m++) {
            double tempTheta[] = new double[K];
            for (int k = 0; k < K; k++) {
                tempTheta[k] = computeTheta(k, m);
            }
            //normalize the sample
            //tempTheta = Utils.normalize(tempTheta, 1.0);
            for (int k = 0; k < K; k++) {
                theta[m][k] += tempTheta[k];
            }
        }
        //average ove all samples
        if (numSamples == totalSamples) {
            for (int m = 0; m < M; m++) {
                for (int k = 0; k < K; k++) {
                    theta[m][k] /= numSamples;
                }
            }
        }
    }

    protected double computeTheta(int k, int m) {
        return nd[m].get(k) + alpha[k];
    }

}
