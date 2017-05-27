package gr.auth.csd.mlkd.mlclassification.labeledlda.subspace;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.set.hash.TIntHashSet;
import gr.auth.csd.mlkd.utils.CmdOption;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Date;
import java.util.HashSet;

/**
 *
 * @author Yannis Papanikolaou
 *
 * Retrieves the n-most relevaqnt documents and creates a subspace of the labels
 * set to be used for LLDA inference
 */
public class MostRelevantTfIdf extends MostRelevant {

    public MostRelevantTfIdf(int n, CmdOption option) {
        super(n, option.testFile, option.trainingFile);

    }

    public MostRelevantTfIdf(int n, String testFile, String trainFile) {
        super(n, testFile, trainFile);
    }

    @Override
    public ArrayList<TObjectDoubleHashMap<String>> mostRelevant() {

        System.out.println(new Date() + " Most relevant labels...");
        trainingFileLabels = gettrainingFileLabels(trainFile);

        trainVectors = readLibSvm(trainFile);
        testVectors = readLibSvm(testFile);
        System.out.println(new Date() + " Finding most relevant...");
        //stores the n-most relevant labels per test document 
        return findMostRelevant();
    }

    @Override
    protected void writeFile(ArrayList<TObjectDoubleHashMap<String>> relevantLabels) {
        ArrayList<TObjectDoubleHashMap<String>> alpha = relevantLabels;
        //System.out.println(alpha);
        try (BufferedWriter output = Files.newBufferedWriter(Paths.get(testFile + ".wlabels"),
                Charset.forName("UTF-8"))) {
            BufferedReader br = new BufferedReader(new FileReader(new File(testFile)));
            String line;
            int doc = 0;
            line = br.readLine();
            while ((line = br.readLine()) != null) {
                String[] splits = line.split(",");
                TObjectDoubleHashMap<String> tags = alpha.get(doc);
                //System.out.println("alpha[0]: "+tags);
                String[] splits2 = splits[splits.length - 1].split(" ");
                StringBuilder sb = new StringBuilder();
                int l = 0;
                for (String tag : tags.keySet()) {
                    sb.append(tag);
                    if (l < tags.size() - 1) {
                        sb.append(",");
                    }
                    l++;
                }

                //features
                for (int feat = 1; feat < splits2.length; feat++) {
                    sb.append(" ");
                    sb.append(splits2[feat]);
                }
                sb.append("\n");
                output.write(sb.toString());
                doc++;
            }
        } catch (IOException ex) {
            Logger.getLogger(MostRelevantTfIdf.class.getName()).log(Level.SEVERE, null, ex);
        }

        try (ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(testFile+".alpha"))) {
            output.writeObject(alpha);
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    protected TIntObjectHashMap<Set<String>> gettrainingFileLabels(String trainFile) {
        TIntObjectHashMap<Set<String>> tfl = new TIntObjectHashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(new File(trainFile)))) {
            String line;
            int d = 0;
            line = br.readLine();
            while ((line = br.readLine()) != null) {
                //System.out.println(doc);
                TIntHashSet tags = this.getTags(line);
                Set<String> docLabels = new HashSet<>();
                TIntIterator it = tags.iterator();
                while (it.hasNext()) {
                    int tag = it.next();
                    docLabels.add(tag + "");
                }
                tfl.put(d, docLabels);
                d++;
            }
        } catch (IOException ex) {
            Logger.getLogger(MostRelevantTfIdf.class.getName()).log(Level.SEVERE, null, ex);
        }
        return tfl;
    }

    private ArrayList<TIntDoubleHashMap> readLibSvm(String trainFile) {
        ArrayList<TIntDoubleHashMap> vector = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(new File(trainFile)))) {
            String line;
            line = br.readLine();
            while ((line = br.readLine()) != null) {
                //System.out.println(doc);
                TIntDoubleHashMap doc = new TIntDoubleHashMap();
                String[] splits = line.split(",");
                String[] splits2 = splits[splits.length - 1].split(" ");
                TIntArrayList features = new TIntArrayList();
                for (int i = 1; i < splits2.length; i++) {
                    String[] featNValue = splits2[i].split(":");
                    int feature = Integer.parseInt(featNValue[0]);
                    double value = Double.parseDouble(featNValue[1]);
                    doc.put(feature, value);
                }
                vector.add(doc);
            }
        } catch (IOException ex) {
            Logger.getLogger(MostRelevantTfIdf.class.getName()).log(Level.SEVERE, null, ex);
        }
        return vector;
    }

    public void evaluate(String testFile, String testFileWLabels) {
        double MaRecall = 0;
        double i = 0, min = 0, max = 0, avg = 0;
        String lineTruth, linePreds;
        try (BufferedReader br = new BufferedReader(new FileReader(new File(testFile)));
                BufferedReader br2 = new BufferedReader(new FileReader(new File(testFileWLabels)))) {
            while ((lineTruth = br.readLine()) != null && (linePreds = br2.readLine()) != null) {
                TIntHashSet predTags = this.getTags(linePreds);
                TIntHashSet truthTags = this.getTags(lineTruth);
                if (i == 0) {
                    min = max = avg = predTags.size();
                } else {
                    if (min > predTags.size()) {
                        min = predTags.size();
                    }
                    if (max < predTags.size()) {
                        max = predTags.size();
                    }
                    avg += predTags.size();
                }
                double rec = recall(truthTags, predTags);
                System.out.println(i + ":" + rec);
                MaRecall += rec;
                i++;
            }
            System.out.println(new Date() + " Macro-recall:" + MaRecall / i + " min = " + min + " max = " + max + " avg = " + avg / i);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(MostRelevantTfIdf.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(MostRelevantTfIdf.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private TIntHashSet getTags(String line) {
        String[] splits = line.split(",");
        TIntHashSet tags = new TIntHashSet();
        for (int i = 0; i < splits.length - 1; i++) {
            tags.add(Integer.parseInt(splits[i]));
        }
        String[] splits2 = splits[splits.length - 1].split(" ");
        if (!splits2[0].isEmpty()) {
            tags.add(Integer.parseInt(splits2[0]));
        }
        return tags;
    }

    private double recall(TIntHashSet truthTags, TIntHashSet predTags) {
        double rec = 0;
        TIntIterator it = truthTags.iterator();
        while (it.hasNext()) {
            if (predTags.contains(it.next())) {
                rec++;
            }
        }
        return rec / truthTags.size();
    }
}
