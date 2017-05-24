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
package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models;

import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.atypon.lda.Dataset;
import gr.auth.csd.mlkd.atypon.utils.Pair;
import gr.auth.csd.mlkd.atypon.utils.Utils;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
public class CGS_pEstimationModel extends Model {

    public TIntDoubleHashMap[] phi_p = null;

    public TIntDoubleHashMap[] getPhi_p() {
        return phi_p;
    }

    public CGS_pEstimationModel(Dataset data, int thread, double beta, String trainedModelName, int threads, int iters, int burnin) {
        super(data, thread, beta, false, trainedModelName, threads, iters, burnin);
        phi_p = new TIntDoubleHashMap[K];
        for (int k = 0; k < K; k++) {
            phi_p[k] = new TIntDoubleHashMap();
        }
    }

    @Override
    public void updateParams(int totalSamples) {
        super.updateParams(totalSamples);
        this.computePhi_p(totalSamples);
        //System.out.println(phi[0]);
    }

    protected TIntDoubleHashMap[] computePhi_p(int totalSamples) {
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
                phi_p[k].adjustOrPutValue(word, tempPhi[k].get(word), tempPhi[k].get(word));
            }
            if (numSamples == totalSamples) {
                phi_p[k] = Utils.normalize(phi_p[k], 1.0);
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

        return phi_p;

    }

    protected void savePhi_p(String modelName) {
        try (ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(modelName + ".phi_p"))) {
            output.writeObject(this.phi_p);
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    public static TIntDoubleHashMap[] readPhi_p(String fi) {
        TIntDoubleHashMap[] p = null;
        try (ObjectInputStream input = new ObjectInputStream(new FileInputStream(fi + "_p"))) {
            p = (TIntDoubleHashMap[]) input.readObject();
        } catch (Exception e) {
            System.out.println(e.getCause());
        }
        System.out.println(fi + "_p loaded: K= " + p.length);
        return p;
    }

    @Override
    public void save(int twords) {
        super.save(twords);
        this.savePhi_p(modelName);
        this.saveTwords_p(modelName+".twords_p", twords);
    }

    public void setPhi_p(TIntDoubleHashMap[] phi_pSum) {
        phi_p = phi_pSum;
    }

    protected void saveTwords_p(String filename, int twords) {
        System.out.println("Saving..");
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"))) {
            for (int k = 0; k < K; k++) {
                ArrayList<Pair> wordsProbsList = new ArrayList<>();
                TIntDoubleIterator it = phi_p[k].iterator();
                while (it.hasNext()) {
                    it.advance();
                    Pair pair = new Pair(it.key(), it.value(), false);
                    wordsProbsList.add(pair);
                }
                //print topic				
                writer.write("Label " + data.getLabel(k + 1) + ":\n");
                Collections.sort(wordsProbsList);
                int iterations = (twords > wordsProbsList.size()) ? wordsProbsList.size() : twords;
                for (int i = 0; i < iterations; i++) {
                    Integer index = (Integer) wordsProbsList.get(i).first;
                    writer.write("\t" + data.getWord(index) + "\t" + wordsProbsList.get(i).second + "\n");
                }
            }
        } catch (IOException e) {
        }
    }
    
}
