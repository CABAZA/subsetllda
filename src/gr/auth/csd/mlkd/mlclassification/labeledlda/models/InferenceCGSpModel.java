package gr.auth.csd.mlkd.mlclassification.labeledlda.models;

import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.mlclassification.labeledlda.Dataset;
import gr.auth.csd.mlkd.utils.Utils;
import java.util.ArrayList;
import java.util.Arrays;



/**
 *
 * @author Yannis Papanikolaou
 */
public class InferenceCGSpModel extends PriorModel {
   
    public InferenceCGSpModel() {
        super();
    }
    
    public InferenceCGSpModel(Dataset data, String trainedModelName, int threads, int iters, int burnin) {
        super(data, trainedModelName, threads, iters, burnin);
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
                    p[k] = (nd[d].get(k) + alpha[k]) * phi[k].get(word);
                }
                nd[d].adjustValue(topic, 1);
                //p = Utils.normalize(p, 1);

                //sum probabilities over the document
                for (int k = 0; k < K; k++) {
                    theta.get(d).adjustOrPutValue(k+1, p[k], p[k]);
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
