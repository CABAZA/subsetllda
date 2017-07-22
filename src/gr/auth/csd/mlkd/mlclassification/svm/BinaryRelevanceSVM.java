package gr.auth.csd.mlkd.mlclassification.svm;

import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.hash.TIntHashSet;
import gr.auth.csd.mlkd.mlclassification.BinaryClassifier;
import gr.auth.csd.mlkd.mlclassification.MLClassifier;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

public class BinaryRelevanceSVM extends MLClassifier {

//    public String modelsDirectory;
//    public TIntObjectHashMap<TreeSet<Integer>> labelValues;
//    protected boolean score = false;
//    int doc2vec;
//    private boolean tuned = false;
    public BinaryRelevanceSVM(String trainingFile, String testFile, int threads) {
        super(trainingFile, testFile, threads);
    }

//    @Override
//    public void train() {
//        File dir = new File(modelsDirectory);
//        if (!dir.exists()) {
//            dir.mkdir();
//        }
//        System.out.println("Training..");
//        labelValues = loadLabels("trainLabels");
//        startThreads(false, null);
//    }
//
//    @Override
//    public double[][] predictInternal() {
////        startThreads(true);
//        predictions = BinaryClassifier.getPredictions(true);
//        return predictions;
//    }

//    public void startThreads(boolean predict) {
//        Thread[] t = new Thread[threads];
//        //System.out.println("creating new binary instances..");
//        for (int i = 0; i < threads; i++) {
//            t[i] = newThread(i, predict, mc);
//            t[i].start();
//        }
//        boolean allDead = false;
//        while (!allDead) {
//            allDead = true;
//            for (int i = 0; i < threads; i++) {
//                if (t[i].isAlive()) {
//                    allDead = false;
//                }
//            }
//        }
//    }

//    protected Thread newThread(int i, boolean predict, TIntHashSet mc) {
//        int numFeatures = 0;
//        //TODO:change it to sth cleaner
//        if(doc2vec==1||doc2vec==2||doc2vec==3) numFeatures+=200;
//        if(!predict) return new Thread(new SVM("train.Libsvm", null, threads, i, 1,
//                numLabels, modelsDirectory, labelValues, (byte) 0, null,
//                numFeatures, score, 0, globalLabels.getSize(), tuned));
//        else return new Thread(new SVM(null, "testFile.libSVM", threads, i, 1, 
//                numLabels, modelsDirectory, null, (byte) 1, mc, 
//                numFeatures, score, CorpusJSON.size(corpus2), globalLabels.getSize(), tuned));
//            
//    }

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

    @Override
    public void train() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void predictInternal() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
