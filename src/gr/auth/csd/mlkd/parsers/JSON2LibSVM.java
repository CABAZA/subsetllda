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
package gr.auth.csd.mlkd.atypon.parsers;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */

import gr.auth.csd.mlkd.atypon.CmdOption;
import gr.auth.csd.mlkd.atypon.features.Wf_Idf;
import gr.auth.csd.mlkd.atypon.preprocessing.Corpus;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Document;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.preprocessing.NGram;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;

public class JSON2LibSVM {

    protected static Dictionary dictionary;
    private final Labels labels;
    private final Wf_Idf fs;
    final boolean tfIdf;

    public JSON2LibSVM(Dictionary d, Labels l, boolean tfIdf) {
        dictionary = d;
        this.labels = l;
        fs = new Wf_Idf(d, false);
        this.tfIdf = tfIdf;
    }

    protected Map<NGram, Integer> nGramFrequencyFromTokenSentences(List<String> lines, int n) {
        Map<NGram, Integer> termFrequency = new HashMap<>();
        for (String line : lines) {
            String[] tokens = line.split(" ");
            for (int i = 0; i < tokens.length + 1 - n; i++) {
                List<String> aList = new ArrayList<>();
                for (int j = 0; j < n; j++) {
                    aList.add(tokens[i + j]);
                }
                NGram ngram = new NGram(aList);
                List<String> list = ngram.getList();
                if (n > 1) {
                    boolean skip = false;
                    for (String token : list) {
                        if (Dictionary.getTokensToIgnore().contains(token)) {
                            skip = true;
                            break;
                        }
                    }
                    if (skip == true) {
                        continue;
                    }
                } else if (Dictionary.getTokensToIgnore().contains(list.get(0))) {
                    continue;
                }
                if (dictionary.getDocumentFrequency().containsKey(ngram)) {
                    if (termFrequency.containsKey(ngram)) {
                        termFrequency.put(ngram, termFrequency.get(ngram) + 1);
                    } else {
                        termFrequency.put(ngram, 1);
                    }
                }
            }
        }
        return termFrequency;
    }

    protected Map<Integer, Double> vectorize(List<String> lines) {
        Map<Integer, Double> vector = new TreeMap<>();
        for (int j = 0; j < dictionary.getNGramSizes().size(); j++) {
            Map<NGram, Integer> termFrequency = nGramFrequencyFromTokenSentences(lines, dictionary.getNGramSizes().get(j));
            Iterator<Map.Entry<NGram, Integer>> entries;
            entries = termFrequency.entrySet().iterator();
            while (entries.hasNext()) {
                Map.Entry<NGram, Integer> entry = entries.next();
                vector.put(dictionary.getId().get(entry.getKey()), entry.getValue().doubleValue());
            }
        }
        return vector;
    }

    protected Map<Integer, Double> vectorize(List<String> lines, boolean lengthNormalization, Document doc) {
        Map<Integer, Double> vector = new TreeMap<>();
        for (int j = 0; j < dictionary.getNGramSizes().size(); j++) {
            Map<NGram, Integer> termFrequency = nGramFrequencyFromTokenSentences(lines, dictionary.getNGramSizes().get(j));
            Iterator<Map.Entry<NGram, Integer>> entries;
            entries = termFrequency.entrySet().iterator();
            while (entries.hasNext()) {
                Map.Entry<NGram, Integer> entry = entries.next();
                vector.put(dictionary.getId().get(entry.getKey()), fs.fsMethod(entry.getKey(), doc, entry.getValue(), labels));
            }
        }

        if (lengthNormalization) {
            vector = normalizeVector(vector);
        }
        return vector;
    }

    protected Map<Integer, Double> normalizeVector(Map<Integer, Double> vector) {
        Collection<Double> weights = vector.values();
        double length = 0;
        for (Double d : weights) {
            length += d * d;
        }
        length = Math.sqrt(length);
        if (length == 0) {
            length = 1;
        }
        Iterator<Map.Entry<Integer, Double>> values = vector.entrySet().iterator();
        while (values.hasNext()) {
            Map.Entry<Integer, Double> entry = values.next();
            entry.setValue(entry.getValue() / length);

        }
        return vector;
    }

    public void parseNWrite(Corpus corpus, String libsvmFilename, Labels corpusLabels) {
        corpus.reset();
        try (BufferedWriter output = Files.newBufferedWriter(Paths.get(libsvmFilename), Charset.forName("UTF-8"))) {

            // read each file in given directory and parse the text as follows
            List<String> lines;
            Document doc;
            int counter = 0;
            while ((doc = corpus.nextDocument()) != null) {
                counter++;
                lines = doc.getContentAsSentencesOfTokens(false);
                Map<Integer, Double> vector = (tfIdf)?vectorize(lines, true, doc):vectorize(lines);
                if (vector != null) {
                    // output features in shell libsvm file
                    StringBuilder sb = new StringBuilder();

                    //labels
                    Set<String> meshTerms = doc.getLabels();
                    HashSet<Integer> labelIds = new HashSet<>();

                    for (String term : meshTerms) {
                        int id = corpusLabels.getIndex(term);
                        if (id != -1) {
                            labelIds.add(id);
                        }

                    }
                    int l = 0;
                    for (Integer id : labelIds) {
                        sb.append(id);
                        if (l < labelIds.size() - 1) {
                            sb.append(",");
                        }
                        l++;
                    }

                    //features
                    Iterator<Map.Entry<Integer, Double>> values = vector.entrySet().iterator();
                    while (values.hasNext()) {
                        Map.Entry<Integer, Double> entry = values.next();
                        if (entry.getValue() != 0) {
                            sb.append(" ");
                            sb.append(entry.getKey() + 1).append(":").append(entry.getValue());
                        }
                    }
                    sb.append("\n");
                    output.write(sb.toString());
                }
            }

        } catch (IOException ex) {
            Logger.getLogger(Dictionary.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void main(String args[]) {
        CmdOption option = new CmdOption(args);
        CorpusJSON in = new CorpusJSON(option.trainingFile);
        Labels labels = Labels.readLabels(option.labels);
        Dictionary dic = Dictionary.readDictionary(option.dictionary);

        String out = option.trainingFile + ".libSVM";
        JSON2LibSVM jl = new JSON2LibSVM(dic, labels, true);
        jl.parseNWrite(in, out, labels);
    }
}
