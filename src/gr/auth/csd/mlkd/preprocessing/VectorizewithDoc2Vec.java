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
package gr.auth.csd.mlkd.atypon.preprocessing;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gr.auth.csd.mlkd.atypon.mlclassification.deeplearning.ParagraphVectorModel;
import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

public class VectorizewithDoc2Vec extends VectorizeJSON {

    protected final ParagraphVectorModel pvm;

    public VectorizewithDoc2Vec(Dictionary d, Labels l, Corpus corpus) {
        super(d, true, l);
        File trainedPV = new File("PV.model");
        if (trainedPV.exists()) {
            pvm = ParagraphVectorModel.read("PV.model");
        } else {
            pvm = new ParagraphVectorModel(corpus, 200);
            pvm.train();
            pvm.save("PV.model");
        }
    }

    @Override
    protected void vectorizeLabeled(Corpus corpus, String libsvmFilename, String labelsFile, boolean perLabel, String metaTrainFileName) {
        corpus.reset();
        try (BufferedWriter output = Files.newBufferedWriter(Paths.get(libsvmFilename), Charset.forName("UTF-8"));
                ObjectOutputStream outLabels = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(labelsFile)))) {
            //System.out.println(new Date() + " Vectorizing labeled data...");

            TIntObjectHashMap<TreeSet<Integer>> labelValues = new TIntObjectHashMap<>();
            TIntArrayList targetValues = new TIntArrayList();
            ArrayList<TIntList> targetValuesPerDoc = new ArrayList<>();

            // read each file in given directory and parse the text as follows
            List<String> lines;
            Document doc;

            int counter = 0;
            while ((doc = corpus.nextDocument()) != null) {
                counter++;

                // output features in shell libsvm file
                StringBuilder sb = new StringBuilder();
                sb.append("0"); //the label value
                //features
                lines = doc.getContentAsSentencesOfTokens(false);
                StringBuilder sb0 = new StringBuilder();
                for (String line : lines) {
                    sb0.append(line).append(" ");
                }
                INDArray inferVector = pvm.doc2vec(sb0.toString(), doc.getLabels());
                Map<Integer, Double> vector = new HashMap<>();
                for (int i = 0; i < inferVector.length(); i++) {
                    double value = inferVector.getDouble(i);
                    if (value != 0) {
                        vector.put(i, value);
                    }
                }
                vector = this.normalizeVector(vector);
                Iterator<Map.Entry<Integer, Double>> it = vector.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry<Integer, Double> next = it.next();
                    sb.append(" ");
                    sb.append(next.getKey() + 1).append(":").append(next.getValue());
                }
                sb.append("\n");
                output.write(sb.toString());

                // record labels
                Set<String> meshTerms = doc.getLabels();
                int cardinality = 0;
                TIntList docLabels = new TIntArrayList();
                for (String term : meshTerms) {
                    //System.out.println("line: " + line);
                    Integer x = labels.getIndex(term);
                    if (x == -1) { //CHANGE in week 5
                        //System.out.println("Label " + line + " not in training corpus");                                
                    } else {
                        docLabels.add(x);
                        cardinality++;
                        TreeSet<Integer> sortedSet;
                        if (labelValues.contains(x)) {
                            sortedSet = labelValues.get(x);
                        } else {
                            sortedSet = new TreeSet<>();
                        }
                        sortedSet.add(counter - 1);
                        labelValues.put(x, sortedSet);
                    }
                }
                //System.out.println("docLabels:" + Arrays.toString(docLabels.toArray()));
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
            Logger.getLogger(VectorizewithDoc2Vec.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    @Override
    public void vectorizeUnlabeled(Corpus aCorpus, String libsvmFilename) {
        try (BufferedWriter output = Files.newBufferedWriter(Paths.get(libsvmFilename), Charset.forName("UTF-8"))) {
            // read each file in given directory and parse the text as follows
            List<String> lines;
            Document doc;
            int counter = 0;
            aCorpus.reset();
            while ((doc = aCorpus.nextDocument()) != null) {
                counter++;

                StringBuilder sb = new StringBuilder();
                sb.append("0"); //the label value
                //features
                lines = doc.getContentAsSentencesOfTokens(false);
                StringBuilder sb0 = new StringBuilder();
                for (String line : lines) {
                    sb0.append(line).append(" ");
                }
                INDArray inferVector = pvm.doc2vec(sb0.toString(), doc.getLabels());
                Map<Integer, Double> vector = new HashMap<>();
                for (int i = 0; i < inferVector.length(); i++) {
                    double value = inferVector.getDouble(i);
                    if (value != 0) {
                        vector.put(i, value);
                    }
                }
                vector = this.normalizeVector(vector);
                Iterator<Map.Entry<Integer, Double>> it = vector.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry<Integer, Double> next = it.next();
                    sb.append(" ");
                    sb.append(next.getKey() + 1).append(":").append(next.getValue());
                }
                sb.append("\n");
                output.write(sb.toString());
            }

        } catch (IOException ex) {
            Logger.getLogger(Dictionary.class.getName()).log(Level.SEVERE, null, ex);
        }
        //System.out.println(new Date() + " Finished vectorizing unlabeled data.");
    }

    @Override
    public ArrayList<TIntDoubleHashMap> vectorizeUnlabeled(Corpus aCorpus) {
        ArrayList<TIntDoubleHashMap> corpusVectors = new ArrayList<>();
        List<String> lines;
        Document doc;
        int counter = 0;
        aCorpus.reset();
        while ((doc = aCorpus.nextDocument()) != null) {
            lines = doc.getContentAsSentencesOfTokens(false);
            StringBuilder sb0 = new StringBuilder();
            for (String line : lines) {
                sb0.append(line).append(" ");
            }
            INDArray inferVector = pvm.doc2vec(sb0.toString(), doc.getLabels());
            Map<Integer, Double> vector = new HashMap<>();
            for (int i = 0; i < inferVector.length(); i++) {
                double value = inferVector.getDouble(i);
                if (value != 0) {
                    vector.put(i, value);
                }
            }
            vector = this.normalizeVector(vector);
            if (vector != null) {
                // output features in shell libsvm file
                TIntDoubleHashMap v = new TIntDoubleHashMap();
                v.putAll(vector);
                corpusVectors.add(v);
            }

        }
        return corpusVectors;
    }

}
