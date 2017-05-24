package gr.auth.csd.mlkd.atypon.preprocessing;

import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
import gr.auth.csd.mlkd.atypon.CmdOption;
import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

public class VectorizeMultiLabelLibSVM extends VectorizeJSON {

    public VectorizeMultiLabelLibSVM(Dictionary d, boolean zoning, Labels labels) {
        super(d, zoning, labels);
    }
    @Override
    protected void vectorizeLabeled(Corpus corpus, String libsvmFilename,
            String labelsFile, boolean perLabel, String metaTrainFileName) {
        ArrayList<String> docMap = new ArrayList<>();
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
            int document = 0;
            while ((doc = corpus.nextDocument()) != null) {
                document++;
                docMap.add(doc.getId());
                lines = doc.getContentAsSentencesOfTokens(false);
                Map<Integer, Double> vector = vectorize(lines, true, doc);
                if (vector != null) {
                    // output features in shell libsvm file
                    Iterator<Map.Entry<Integer, Double>> values = vector.entrySet().iterator();
                    StringBuilder sb = new StringBuilder();
                    HashSet<Integer> labelIds = new HashSet<>();

                    //System.out.println("line: " + line);
                    //System.out.println("docLabels:" + Arrays.toString(docLabels.toArray()));
                    //labels
                    Set<String> meshTerms = doc.getLabels();
                    int cardinality = 0;
                    TIntList docLabels = new TIntArrayList();
                    for (String term : meshTerms) {
                        int id = labels.getIndex(term);
                        if (id != -1) {
                            labelIds.add(id);
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

                    }

                    targetValuesPerDoc.add(docLabels);
                    targetValues.add(cardinality);

                    //features
                    int l = 0;
                    for (Integer id : labelIds) {
                        sb.append(id-1);
                        if (l < labelIds.size() - 1) {
                            sb.append(",");
                        }
                        l++;
                    }
                    while (values.hasNext()) {
                        Map.Entry<Integer, Double> entry = values.next();
                        if (entry.getValue() != 0) {
                            sb.append(" ");
                            sb.append(entry.getKey() + 1).append(":").append(String.format(Locale.US, "%.6f", entry.getValue()));
                        }
                    }
                    sb.append("\n");
                    output.write(sb.toString());
                }
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
            Logger.getLogger(VectorizeMultiLabelLibSVM.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
        try {
            StringBuilder sb = new StringBuilder();
            for(String doc:docMap) {
                sb.append(doc).append("\n");
            }
            Files.write(Paths.get("docMap.txt"), sb.toString().getBytes());
        } catch (IOException ex) {
            Logger.getLogger(Labels.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
    public static void main (String[] args) {
        Dictionary dic;
        CmdOption option = new CmdOption(args);
        CorpusJSON corpus = new CorpusJSON(option.trainingFile);
        CorpusJSON corpus2 = new CorpusJSON(option.testFile);
        dic = new Dictionary(corpus, 5, 10000, 10, 5000);
        Labels labels = new Labels(corpus);
        //labels.writeLabels(option.labels);
        labels.writeToFile("labels");
        dic.writeDictionary(option.dictionary);
        VectorizeMultiLabelLibSVM vectorize = new VectorizeMultiLabelLibSVM(dic, true, labels);
        vectorize.vectorizeTrain(corpus, "tr", "trainLabels", "metaTrainLabels");
        vectorize.vectorizeTrain(corpus2, "tst", "trainLabels", "metaTrainLabels");
    }
}
