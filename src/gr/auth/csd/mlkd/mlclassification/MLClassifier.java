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
    protected ArrayList<TreeMap<Integer, Double>> predictions;
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

    public void predict() {
        predictInternal();
        savePredictions();

    }

    public abstract void train();

    public abstract void predictInternal();

    public void savePredictions() {
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(predictionsFilename)))) {
            for (TreeMap<Integer, Double> p1 : predictions) {
                StringBuilder sb = new StringBuilder();
                Iterator<Map.Entry<Integer, Double>> it = p1.entrySet().iterator();
                int i=0;
                while(it.hasNext()) {
                    Map.Entry<Integer, Double> next = it.next();
                    sb.append(next.getKey()).append(":").append(next.getValue());
                    if(i<p1.size()-1) sb.append(" ");
                    else sb.append("\n");
                }  
                writer.write(sb.toString());
            }

        } catch (Exception ex) {
            Logger.getLogger(MLClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public ArrayList<TreeMap<Integer, Double>> getPredictions() {
        return this.predictions;
    }

    public void setPredictions(ArrayList<TreeMap<Integer, Double>> predictions) {
        this.predictions = predictions;
    }

}
