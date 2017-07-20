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

import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Yannis Papanikolaou
 */
public abstract class MLClassifier {

    protected int threads;
    protected ArrayList<TIntDoubleHashMap> predictions;
    public String testFile;
    protected String trainingFile;
    protected int offset = 0;
    protected String predictionsFilename = "predictions";

    public MLClassifier(String trainingFile, String testFile, int threads) {
        this.trainingFile = trainingFile;
        this.testFile = testFile;
        this.threads = threads;
    }

    public void predict() {
        predictInternal();
        savePredictions();

    }

    public abstract void train();

    public abstract void predictInternal();

    public void savePredictions() {
        int nrLabels = 0;
        for (TIntDoubleHashMap p1 : predictions) {
            TIntIterator it = p1.keySet().iterator();
            while(it.hasNext()) {
                int next = it.next();
                if(next>nrLabels) nrLabels = next;
            }
        }
        
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(predictionsFilename)))) {
            writer.write(this.getPredictions().size()+" "+(nrLabels+1)+"\n");
            for (TIntDoubleHashMap p1 : predictions) {
                StringBuilder sb = new StringBuilder();
                TIntDoubleIterator it = p1.iterator();
                TIntDoubleHashMap nonzero = new TIntDoubleHashMap();
                while(it.hasNext()) {
                    it.advance();
                    if(it.value()!=0) nonzero.put(it.key(), it.value());
                }
                it = nonzero.iterator();
                int i=0;
                while(it.hasNext()) {
                    it.advance();
                    sb.append(it.key()).append(":").append(it.value());
                    if(i<nonzero.size()-1) sb.append(" ");
                    else sb.append("\n");
                    i++;
                }
                writer.write(sb.toString());
            }

        } catch (Exception ex) {
            Logger.getLogger(MLClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public ArrayList<TIntDoubleHashMap> getPredictions() {
        return this.predictions;
    }

    public void setPredictions(ArrayList<TIntDoubleHashMap> predictions) {
        this.predictions = predictions;
    }

}
