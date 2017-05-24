package gr.auth.csd.mlkd.atypon.mlclassification.svm;

import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.hash.TIntHashSet;
import gr.auth.csd.mlkd.atypon.mlclassification.BinaryClassifier;
import gr.auth.csd.mlkd.atypon.mlclassification.MLClassifier;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizeJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizewithAutoEncoder;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizewithDoc2Vec;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizewithDoc2VecAndTfIdf;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

public class BinaryRelevanceSVM extends MLClassifier {

    protected static VectorizeJSON vectorize;
    public String modelsDirectory;
    public TIntObjectHashMap<TreeSet<Integer>> labelValues;
    protected boolean score = false;
    int doc2vec;
    private boolean tuned = false;

    public BinaryRelevanceSVM(String trainingFile, String testFile, String dic, 
            String labels, String modelsDirectory, int threads, int doc2vec, boolean tuned) {
        
        this((dic == null)? null:Dictionary.readDictionary(dic), (labels == null)?null:Labels.readLabels(labels),
                ((trainingFile != null) ? new CorpusJSON(trainingFile) : null), 
                ((testFile != null) ? new CorpusJSON(testFile) : null), modelsDirectory, 
                threads, doc2vec, tuned);
    }

    public BinaryRelevanceSVM(String trainingFile, String testFile, Dictionary dictionary, 
            Labels labels, String modelsDirectory, int threads, int doc2vec, boolean tuned) {
        
        this(dictionary, labels, ((trainingFile != null) ? new CorpusJSON(trainingFile) : null),
                ((testFile != null) ? new CorpusJSON(testFile) : null), modelsDirectory, threads, doc2vec, tuned);
    }

    public BinaryRelevanceSVM(Dictionary dictionary, Labels labels, CorpusJSON trainingCorpus,
            CorpusJSON testCorpus, String modelsDirectory, int threads, int doc2vec, boolean tuned) {
        super(dictionary, labels, trainingCorpus, testCorpus, threads);
        vectorize = (doc2vec==1)?new VectorizewithDoc2Vec(dictionary, labels, trainingCorpus):
                (doc2vec==2)?new VectorizewithAutoEncoder(dictionary, labels, trainingCorpus):
                (doc2vec==3)?new VectorizewithDoc2VecAndTfIdf(dictionary, labels, trainingCorpus):
                new VectorizeJSON(dictionary, true, labels);
        this.doc2vec = doc2vec;
        this.modelsDirectory = modelsDirectory;
        this.tuned = tuned;
    }

    @Override
    public void train() {
        File dir = new File(modelsDirectory);
        if (!dir.exists()) {
            dir.mkdir();
        }
        vectorize.vectorizeTrain(corpus, "train.Libsvm", "trainLabels", "metaTrainLabels");
//        if (corpus2 != null) {
//            vectorize.vectorizeUnlabeled(corpus2, "testFile.libSVM");
//        }
        System.out.println("Training..");
        labelValues = loadLabels("trainLabels");
        startThreads(false, null);
//        if (corpus2 != null) {
//            predictions = BinaryClassifier.getPredictions();
//        }
    }

    @Override
    public double[][] predictInternal(TIntHashSet mc) {
        
        vectorize.vectorizeUnlabeled(corpus2, "testFile.libSVM");
        startThreads(true, mc);
        predictions = BinaryClassifier.getPredictions(true);
        return predictions;
    }

    public void startThreads(boolean predict, TIntHashSet mc) {
        Thread[] t = new Thread[threads];
        //System.out.println("creating new binary instances..");
        for (int i = 0; i < threads; i++) {
            t[i] = newThread(i, predict, mc);
            t[i].start();
        }
        boolean allDead = false;
        while (!allDead) {
            allDead = true;
            for (int i = 0; i < threads; i++) {
                if (t[i].isAlive()) {
                    allDead = false;
                }
            }
        }
    }

    protected Thread newThread(int i, boolean predict, TIntHashSet mc) {
        int numFeatures = dictionary.getId().size();
        //TODO:change it to sth cleaner
        if(doc2vec==1||doc2vec==2||doc2vec==3) numFeatures+=200;
        if(!predict) return new Thread(new SVM("train.Libsvm", null, threads, i, 1,
                numLabels, modelsDirectory, labelValues, (byte) 0, null,
                numFeatures, score, 0, globalLabels.getSize(), tuned));
        else return new Thread(new SVM(null, "testFile.libSVM", threads, i, 1, 
                numLabels, modelsDirectory, null, (byte) 1, mc, 
                numFeatures, score, CorpusJSON.size(corpus2), globalLabels.getSize(), tuned));
            
    }

    private TIntObjectHashMap<TreeSet<Integer>> loadLabels(String filenameLabels) {
        try (ObjectInputStream input = new ObjectInputStream(new BufferedInputStream(new FileInputStream(filenameLabels)))) {
            //System.out.println(new Date() + " loading training labels...");
            TIntObjectHashMap<TreeSet<Integer>> labelValuesTemp = (TIntObjectHashMap<TreeSet<Integer>>) input.readObject();
            //System.out.println(new Date() + " Finished.");
            return labelValuesTemp;
        } catch (ClassNotFoundException | IOException e) {
            Logger.getLogger(BinaryRelevanceSVM.class.getName()).log(Level.SEVERE, null, e);
            return null;
        }
    }
}
