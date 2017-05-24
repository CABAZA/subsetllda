package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace;

import gnu.trove.iterator.TDoubleIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.set.hash.THashSet;
import gr.auth.csd.mlkd.atypon.CmdOption;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Document;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import gr.auth.csd.mlkd.atypon.parsers.Parser;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizeJSON;
import java.util.Date;
import org.codehaus.jackson.JsonEncoding;
import org.codehaus.jackson.JsonFactory;
import org.codehaus.jackson.JsonGenerator;

/**
 *
 * @author Yannis Papanikolaou
 *
 * Retrieves the n-most relevaqnt documents and creates a subspace of the labels
 * set to be used for LLDA inference
 */
public class MostRelevantTfIdf extends MostRelevant {

    final Dictionary dic;
    final Labels ls;
    private double norm1;
    private double norm2;

    public MostRelevantTfIdf(int n,  CmdOption option) {
        super(n, option.testFile, option.trainingFile);
                CorpusJSON corpus = new CorpusJSON(option.trainingFile);
        if ((new File(option.dictionary)).exists()) {
            dic = Dictionary.readDictionary(option.dictionary);
        } else {
            dic = new Dictionary(corpus, option.lowUnigrams, option.highUnigrams, option.lowBigrams, option.highBigrams);
        }

        if ((new File(option.labels)).exists()) {
            this.ls = Labels.readLabels(option.labels);
        } else {
            this.ls = new Labels(corpus);
        }
    }

    public MostRelevantTfIdf(int n, String testFile, String trainFile, String dic, String ls) {
        super(n, testFile, trainFile);
        this.dic = Dictionary.readDictionary(dic);
        this.ls = Labels.readLabels(ls);
    }

    public MostRelevantTfIdf(int n, String testFile, String trainFile, Dictionary dic, Labels ls) {
        super(n, testFile, trainFile);;
        this.dic = dic;
        this.ls = ls;
    }

    @Override
    public ArrayList<TObjectDoubleHashMap<String>> mostRelevant() {

        System.out.println(new Date() + " Most relevant labels...");
        VectorizeJSON vectorize = new VectorizeJSON(dic, true, ls);
        CorpusJSON training = new CorpusJSON(trainFile);
        CorpusJSON test = new CorpusJSON(testFile);
        trainingFileLabels = gettrainingFileLabels(training);

        trainVectors = vectorize.vectorizeUnlabeled(training);
        testVectors = vectorize.vectorizeUnlabeled(test);
        System.out.println(new Date() + " Finding most relevant...");
        //stores the n-most relevant labels per test document 
        return findMostRelevant();
    }

    private double norm(TIntDoubleHashMap a) {
        double norm = 0;
        for (double value : a.values()) {
            norm += value * value;
        }
        return Math.sqrt(norm);
    }

    @Override
    protected void writeFile(ArrayList<TObjectDoubleHashMap<String>> relevantLabels) {
        CorpusJSON c = new CorpusJSON(testFile);
        ArrayList<TObjectDoubleHashMap<String>> alpha = relevantLabels;
        JsonFactory jfactory = new JsonFactory();
        Document doc;
        try (JsonGenerator jGenerator = jfactory.createJsonGenerator(new File(testFile + ".wlabels"), JsonEncoding.UTF8)) {
            jGenerator.writeStartObject();
            jGenerator.writeFieldName("documents");
            jGenerator.writeStartArray();
            c.reset();
            int d = 0;
            while ((doc = c.nextDocument()) != null) {
                THashSet<String> ls = new THashSet<>();
                ls.addAll(relevantLabels.get(d).keySet());
                int in = 0;
                for (String label : doc.getLabels()) {
                    if (ls.contains(label)) {
                        in++;
                    }
                }
                //System.out.println("In the set:" + in + " percentage: " + in * 1.0 / doc.getLabels().size() + " total:" + ls.size()+" true labelset:"+doc.getLabels().size());

//                if (in * 1.0 / doc.getLabels().size() >= 0.8) {
//                    success++;
//                }
                Parser.write(jGenerator, doc.getId(), doc.getTitle(), doc.getAbs(),
                        Integer.toString(doc.getYear()), doc.getJournal(), ls, doc.getBody());
                d++;
            }
            //System.out.println(success);
            jGenerator.writeEndArray();
            jGenerator.writeEndObject();
        } catch (IOException ex) {
            Logger.getLogger(MostRelevantTfIdf.class.getName()).log(Level.SEVERE, null, ex);
        }

        try (ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream("alpha"+this.testFile))) {
            output.writeObject(alpha);
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    protected TIntObjectHashMap<Set<String>> gettrainingFileLabels(CorpusJSON c) {
        TIntObjectHashMap<Set<String>> tfl = new TIntObjectHashMap<>();
        c.reset();
        Document doc;
        int d = 0;
        while ((doc = c.nextDocument()) != null) {
            tfl.put(d, doc.getLabels());
            d++;
        }
        return tfl;
    }

    public static void evaluate(String testFile, String testFileWLabels) {
        CorpusJSON c = new CorpusJSON(testFile);
        CorpusJSON cwLabels = new CorpusJSON(testFileWLabels);
        c.reset();
        cwLabels.reset();
        Document doc, docwLabels;
        double MaRecall = 0;
        double i = 0, min = 0, max = 0, avg = 0;
        while ((doc = c.nextDocument()) != null) {
            docwLabels = cwLabels.nextDocument();
            if (i == 0) {
                min = max = avg = docwLabels.getLabels().size();
            } else {
                if (min > docwLabels.getLabels().size()) {
                    min = docwLabels.getLabels().size();
                }
                if (max < docwLabels.getLabels().size()) {
                    max = docwLabels.getLabels().size();
                }
                avg += docwLabels.getLabels().size();
            }
            double rec = recall(doc.getLabels(), docwLabels.getLabels());
            //System.out.println(doc.getId() + ":" + rec + " " + docwLabels.getLabels().size());
            MaRecall += rec;
            i++;
        }
        System.out.println(new Date() + " Macro-recall:" + MaRecall / i + " min = " + min + " max = " + max + " avg = " + avg / i);
    }

    private static double recall(THashSet<String> labels, THashSet<String> labels0) {
        double rec = 0;
        for (String label : labels) {
            if (labels0.contains(label)) {
                rec++;
            }
        }
        return rec / labels.size();
    }

    private void cleanRareLabels(ArrayList<TObjectDoubleHashMap<String>> labels, int threshold) {
        for (TObjectDoubleHashMap<String> l : labels) {
            TDoubleIterator iterator = l.valueCollection().iterator();
            while (iterator.hasNext()) {
                if (iterator.next() < threshold) {
                    iterator.remove();
                }
            }
        }
    }


}
