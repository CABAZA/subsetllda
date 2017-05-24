package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models;

import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.atypon.lda.Dataset;
import gr.auth.csd.mlkd.atypon.lda.DatasetTfIdf;



/**
 *
 * @author Yannis Papanikolaou
 */
public class PriorModelTfIdf extends ModelTfIdf {

    public PriorModelTfIdf() {
        super();
    }

    public PriorModelTfIdf(Dataset data, String trainedModelName, int threads, int iters, int burnin) {
        super(data, 1, 0.01, true, trainedModelName, threads, iters, burnin);
    }

    @Override
    public void initAlpha() {
        alpha = new double[K];
        double sumOfFrequencies = 0;
        TIntDoubleHashMap labels = ((DatasetTfIdf)data).getLabels();
        TIntDoubleIterator it = labels.iterator();
        while(it.hasNext()) {
            it.advance();
            sumOfFrequencies += it.value();
        }
        
        it = labels.iterator();
        while(it.hasNext()) {
            it.advance();
            int k = it.key();
            double frequency = it.value();
            alpha[k-1] = 50.0 * frequency / sumOfFrequencies + 30.0 / K;
        }
    }

}
