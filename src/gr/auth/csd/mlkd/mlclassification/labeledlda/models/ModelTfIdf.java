package gr.auth.csd.mlkd.mlclassification.labeledlda.models;

import gr.auth.csd.mlkd.mlclassification.labeledlda.Dataset;


public class ModelTfIdf extends Model {

    public ModelTfIdf() {
    }


    public ModelTfIdf(Dataset data, int thread, double beta, boolean inf, String trainedModelName, int threads, int iters, int burnin) {
        super(data, thread, beta, inf, trainedModelName, threads, iters, burnin);
    }


    @Override
    public void setZInitially(int m, int word, int topic) {
        z[m].put(word, topic);
        double tfIdfValue = data.getDocs().get(m).getTfIdfFeatures().get(word);
        nd[m].adjustOrPutValue(topic, tfIdfValue, tfIdfValue);
        if (!inference) {
            nw[topic].adjustOrPutValue(word, tfIdfValue, tfIdfValue);
            nwsum[topic]+=tfIdfValue; // total number of words assigned to topic j 
        }
    }

    @Override
    public void removeZi(int m, int w, int topic) {
        double tfIdfValue = data.getDocs().get(m).getTfIdfFeatures().get(w);
        nd[m].adjustValue(topic, -tfIdfValue);
//        if (nd[m].get(topic) <= 0) {
//            nd[m].remove(topic);
//        }
        if (!inference) {
            nwsum[topic] -= tfIdfValue;
            nw[topic].adjustValue(w, -tfIdfValue);
//            if (nw[topic].get(w) == 0) {
//                nw[topic].remove(w);
//            }
        }
    }

    @Override
    public void addZi(int m, int w, int topic) {
        double tfIdfValue = data.getDocs().get(m).getTfIdfFeatures().get(w);
        nd[m].adjustOrPutValue(topic,tfIdfValue , tfIdfValue);
        if (!inference) {
            nw[topic].adjustOrPutValue(w, tfIdfValue, tfIdfValue);
            nwsum[topic] += tfIdfValue;
        }
    }

}
