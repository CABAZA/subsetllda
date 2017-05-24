package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models;

import gr.auth.csd.mlkd.atypon.lda.Dataset;
import gr.auth.csd.mlkd.atypon.lda.DatasetBoW;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;



/**
 *
 * @author Yannis Papanikolaou
 */
public class PriorModel extends Model {

    private final Labels labels;

    public PriorModel(Dataset data, String trainedModelName, int threads, int iters, int burnin) {
        super(data, 1, 0.01, true, trainedModelName, threads, iters, burnin);
        labels = ((DatasetBoW)data).getLabels();
    }

    @Override
    public void initAlpha() {
        alpha = new double[K];
        double sumOfFrequencies = 0;
        for (int i = 1; i < labels.getSize(); i++) {
            String label = labels.getLabel(i);
            sumOfFrequencies += labels.getPositiveInstances().get(label);
        }
        for (int k = 0; k < K; k++) {
            //convert k+1 index to the Labels object index in order to retrieve label's frequency
            String label = labels.getLabel(k + 1);
            double frequency = labels.getPositiveInstances().get(label);
            alpha[k] = 50.0 * frequency / sumOfFrequencies + 30.0 / K;
        }
    }

}
