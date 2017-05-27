/*
 * Copyright (C) 2015 Yannis Papanikolaou <ypapanik@csd.auth.gr>
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
package gr.auth.csd.mlkd.preprocessing;

import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Date;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
public class VectorizeLibSvm implements Vectorize {

    protected void vectorizeLabeled(String in, String out,
            String labelsFile, boolean perLabel, String metaTrainFileName) {
        try (BufferedWriter output = Files.newBufferedWriter(Paths.get(out),
                Charset.forName("UTF-8"));
                ObjectOutputStream outLabels = new ObjectOutputStream(
                        new BufferedOutputStream(new FileOutputStream(labelsFile)))) {
            BufferedReader br = new BufferedReader(new FileReader(in));
            //System.out.println(new Date() + " Vectorizing labeled data...");
            TIntObjectHashMap<TreeSet<Integer>> labelValues = new TIntObjectHashMap<>();
            TIntArrayList targetValues = new TIntArrayList();
            ArrayList<TIntList> targetValuesPerDoc = new ArrayList<>();
            // read each file in given directory and parse the text as follows

            String line;
            int document = 0;
            line = br.readLine();
            while ((line = br.readLine()) != null) {
                document++;
                String[] splits = line.split(",");
                ArrayList<String> labels = new ArrayList<>();
                for (int i = 0; i < splits.length - 1; i++) {
                    labels.add(splits[i]);
                }
                String[] splits2 = splits[splits.length - 1].split(" ");
                labels.add(splits2[0]);
                StringBuilder sb = new StringBuilder();
                sb.append("0 ");
                for (int i = 1; i < splits2.length - 1; i++) {
                    String[] featNValue = splits2[i].split(":");
                    sb.append(Integer.parseInt(featNValue[0])+1).append(":").append(featNValue[1]).append(" ");
                }
                String[] featNValue = splits2[splits2.length - 1].split(":");
                sb.append(Integer.parseInt(featNValue[0])+1).append(":").append(featNValue[1]).append("\n");
                //System.out.println(sb.toString());
                output.write(sb.toString());

                int cardinality = 0;
                TIntList docLabels = new TIntArrayList();
                for (String term : labels) {
                    int id = Integer.parseInt(term);
                    docLabels.add(id);
                    cardinality++;
                    TreeSet<Integer> sortedSet;
                    if (labelValues.contains(id)) {
                        sortedSet = labelValues.get(id);
                    } else {
                        sortedSet = new TreeSet<>();
                    }
                    sortedSet.add(document - 1);
                    labelValues.put(id, sortedSet);
                }
                targetValuesPerDoc.add(docLabels);
                targetValues.add(cardinality);

            }

            if (perLabel) {
                outLabels.writeObject(labelValues);
                try (ObjectOutputStream metaTrain = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(metaTrainFileName)))) {
                    metaTrain.writeObject(targetValues);
                }

            } else {
                outLabels.writeObject(targetValuesPerDoc);
            }
        } catch (IOException ex) {
            Logger.getLogger(VectorizeLibSvm.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void vectorizeTrain(String in, String out) {
        this.vectorizeLabeled(in, out, "trainLabels", true, "metaTrainLabels");
    }

    public void vectorizeUnlabeled(String in, String out) {
        try (BufferedWriter output = Files.newBufferedWriter(Paths.get(out),
                Charset.forName("UTF-8"));) {
            BufferedReader br = new BufferedReader(new FileReader(in));

            // read each file in given directory and parse the text as follows
            String line;
            int document = 0;
            while ((line = br.readLine()) != null) {
                document++;
                String[] splits = line.split(",");
                String[] splits2 = splits[splits.length - 1].split(" ");
                StringBuilder sb = new StringBuilder();
                sb.append("0 ");
                for (int i = 1; i < splits2.length - 1; i++) {
                    String[] featNValue = splits2[i].split(":");
                    sb.append(Integer.parseInt(featNValue[0])+1).append(":").append(featNValue[1]).append(" ");
                }
                String[] featNValue = splits2[splits2.length - 1].split(":");
                sb.append(Integer.parseInt(featNValue[0])+1).append(":").append(featNValue[1]).append("\n");
                output.write(sb.toString());

            }
        } catch (IOException ex) {
            Logger.getLogger(VectorizeLibSvm.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}
