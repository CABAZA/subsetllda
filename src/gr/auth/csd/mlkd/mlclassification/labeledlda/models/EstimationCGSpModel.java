/*
 * Copyright (C) 2016 Yannis Papanikolaou <ypapanik@csd.auth.gr>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package gr.auth.csd.mlkd.mlclassification.labeledlda.models;

import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.mlclassification.labeledlda.Dataset;
import gr.auth.csd.mlkd.utils.Utils;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
public class EstimationCGSpModel extends ModelTfIdf {


    public EstimationCGSpModel(Dataset data, int thread, double beta, String trainedModelName, int threads, int iters, int burnin) {
        super(data, thread, beta, false, trainedModelName, threads, iters, burnin);
    }

    @Override
    public void updateParams(int totalSamples) {
        super.updateParams(totalSamples);
    }

    @Override
    protected TIntDoubleHashMap[] computePhi(int totalSamples) {
        TIntDoubleHashMap[] tempPhi = new TIntDoubleHashMap[K];
        for (int k = 0; k < K; k++) {
            tempPhi[k] = new TIntDoubleHashMap();
        }
        //accumulate probabilities
        for (int d = 0; d < M; d++) {
            int[] labels = data.getDocs().get(d).getLabels();
            TIntArrayList words = data.getDocs().get(d).getWords();
            for (int w = 0; w < words.size(); w++) {
                double[] p = new double[labels.length];
                int word = data.getDocs().get(d).getWords().get(w);
                int topic = z[d].get(w);
                nd[d].adjustValue(topic, -1);
                for (int k = 0; k < labels.length; k++) {
                    int label = labels[k] - 1;
                    p[k] = this.probability(word, label, d);
                }
                nd[d].adjustValue(topic, 1);
                p = Utils.normalize(p, 1);
                for (int k = 0; k < labels.length; k++) {
                    int label = labels[k] - 1;
                    tempPhi[label].adjustOrPutValue(word, p[k], p[k]);
                }
            }
        }

        for (int k = 0; k < K; k++) {
            TIntDoubleIterator iterator = tempPhi[k].iterator();
            while (iterator.hasNext()) {
                iterator.advance();
                iterator.setValue(iterator.value()+beta);
            }
            tempPhi[k] = Utils.normalize(tempPhi[k], 1.0);

            iterator = tempPhi[k].iterator();
            while (iterator.hasNext()) {
                iterator.advance();
                int word = iterator.key();
                phi[k].adjustOrPutValue(word, tempPhi[k].get(word), tempPhi[k].get(word));
            }
            if (numSamples == totalSamples) {
                phi[k] = Utils.normalize(phi[k], 1.0);
            }
        }
//
//        for (int k = 0; k < K; k++) {
//            if (numSamples == totalSamples) {
//                TIntDoubleIterator it = phi_p[k].iterator();
//                while(it.hasNext()) {
//                    it.advance();
//                    phi_p[k].adjustValue(it.key(), beta);
//                }
//                phi_p[k] = Utils.normalize(phi_p[k], 1.0);
//            }
//        }

        return phi;

    }   
}
