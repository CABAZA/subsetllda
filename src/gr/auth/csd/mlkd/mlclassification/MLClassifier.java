/*
 * Copyright (C) 2015 Yannis Papanikolaou
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
package gr.auth.csd.mlkd.mlclassification;

import gnu.trove.iterator.TObjectDoubleIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.set.hash.TIntHashSet;
import gr.auth.csd.mlkd.utils.Utils;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Yannis Papanikolaou
 */
public abstract class MLClassifier {

    protected int threads;
    protected double[][] predictions;
    protected int numLabels = 0;
    public String testFile;
    protected String trainingFile;
    protected int offset = 0;
    protected String predictionsFilename = "predictions";

    public MLClassifier(String trainingFile, String testFile, 
            int nLabels, int threads) {
        this.trainingFile = trainingFile;
        this.testFile = testFile;
        numLabels = nLabels;
        this.threads = threads;
    }

    public void predict(TIntHashSet mc) {
        predictInternal(mc);
        savePredictions();

    }

    public abstract void train();

    public abstract double[][] predictInternal(TIntHashSet mc);

    protected void writeProbs(TreeMap<String, TObjectDoubleHashMap<String>> probMap, String scorestxt) {
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(scorestxt)))) {
            Iterator<Map.Entry<String, TObjectDoubleHashMap<String>>> it = probMap.entrySet().iterator();
            while (it.hasNext()) {
                Map.Entry<String, TObjectDoubleHashMap<String>> next = it.next();
                //System.out.println(next.getValue());
                TObjectDoubleIterator<String> it2 = next.getValue().iterator();
                TreeMap<Integer, Double> ordered = new TreeMap<>();
                while (it2.hasNext()) {
                    it2.advance();
                    ordered.put(Integer.parseInt(it2.key()), it2.value());
                }
                StringBuilder sb = new StringBuilder();
                int i = 0;
                Iterator<Map.Entry<Integer, Double>> it3 = ordered.entrySet().iterator();
                while (it3.hasNext()) {
                    Map.Entry<Integer, Double> n = it3.next();
                    sb.append(n.getKey() - 1).append(":").append(Math.round(n.getValue() * 1000000.0) / 1000000.0);
                    if (i < ordered.size() - 1) {
                        sb.append(" ");
                    }
                    i++;
                }
                sb.append("\n");
                writer.write(sb.toString());
            }
        } catch (Exception ex) {
            Logger.getLogger(MLClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void savePredictions() {
        double p2[][] = new double[predictions.length][];
        for (int doc = 0; doc < predictions.length; doc++) {
            p2[doc] = Arrays.copyOf(predictions[doc], predictions[0].length);
        }
        ArrayList<TIntDoubleHashMap> p = new ArrayList<>();
        for (int doc = 0; doc < p2.length; doc++) {
            TIntDoubleHashMap preds = new TIntDoubleHashMap();
            if (100 > predictions[0].length) {
                for (int k = 0; k < predictions[0].length; k++) {
                    if (p2[doc][k] != 0) {
                        preds.put(k, p2[doc][k]);
                    }
                }
            } else {
                for (int k = 0; k < 100; k++) {
                    int label = Utils.maxIndex(p2[doc]);
                    if (p2[doc][label] != 0) {
                        preds.put(label, p2[doc][label]);
                    }
                    p2[doc][label] = -1;
                }
            }
            p.add(doc, preds);
        }
        Utils.writeObject(p, predictionsFilename);
    }
}
