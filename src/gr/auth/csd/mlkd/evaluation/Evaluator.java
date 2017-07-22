/*
 * Copyright (C) 2015 user
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
package gr.auth.csd.mlkd.evaluation;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gr.auth.csd.mlkd.utils.Utils;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Yannis Papanikolaou
 */
public abstract class Evaluator {

    protected String filenamePredictions;
    protected HashMap<Integer, TreeSet<Integer>> bipartitions = new HashMap<>();
    TIntObjectHashMap<TreeSet<Integer>> truth;

    public abstract void evaluate();

    protected void readBipartitions(int rcut) {
        try (BufferedReader br = new BufferedReader(new FileReader(filenamePredictions))) {
            String line;
            int d = 0;
            line = br.readLine();
            while ((line = br.readLine()) != null) {
                TIntDoubleHashMap predictionsPerDoc = new TIntDoubleHashMap();
                //read
                String[] labelscores = line.split(" ");
                //if(d==0) System.out.println(d+" "+Arrays.toString(labelscores));
                for (String ls : labelscores) {
                    String[] split = ls.split(":");
                    int l = Integer.parseInt(split[0]);
                    double score = Double.parseDouble(split[1]);
                    predictionsPerDoc.put(l, score);
                }
                //write
                TreeSet<Integer> preds = new TreeSet<>();
                for (int i = 0; i < rcut; i++) {
                    int maxIndex = Utils.maxIndex(predictionsPerDoc);
                    //if(d==0) System.out.println(maxIndex+" "+predictionsPerDoc.get(maxIndex));
                    predictionsPerDoc.put(maxIndex, Double.MIN_VALUE);
                    preds.add(maxIndex);
                }
                //if(d==0) System.out.println(d+" "+preds);
                bipartitions.put(d, preds);
                d++;
            }
        } catch (IOException ex) {
            Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    protected TIntObjectHashMap<TreeSet<Integer>> readTruth(String folderTest) {
        TIntObjectHashMap<TreeSet<Integer>> tru = new TIntObjectHashMap<>();
        try (final BufferedReader reader = new BufferedReader(new FileReader(folderTest))) {
            String line;
            int doc = 0;
            line = reader.readLine();
            while ((line = reader.readLine()) != null) {
                TreeSet<Integer> truth = new TreeSet<>();
                String[] split = line.split(" ");
                String[] tags = split[0].split(",");
                if (tags.length != 0&&!tags[0].isEmpty()) {
                    for (String label : tags) {
                        truth.add(Integer.parseInt(label));
                    }
                }
                tru.put(doc, truth);
                doc++;
            }
        } catch (IOException ex) {
            Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null, ex);
        }
        return tru;
    }
}
