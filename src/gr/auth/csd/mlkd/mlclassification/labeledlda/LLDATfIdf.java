package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda;

import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.iterator.TObjectDoubleIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.set.hash.THashSet;
import gnu.trove.set.hash.TIntHashSet;
import gr.auth.csd.mlkd.atypon.LLDACmdOption;
import gr.auth.csd.mlkd.atypon.lda.DatasetTfIdf;
import gr.auth.csd.mlkd.atypon.mlclassification.MLClassifier;

import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.CGS_pModelTfIdf;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.Model;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.ModelTfIdf;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.OnlyCGS_pPriorModelTfIdf;
import gr.auth.csd.mlkd.atypon.mlclassification.svm.MetaModel;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizeLibSvm;
import gr.auth.csd.mlkd.atypon.utils.Utils;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;

public class LLDATfIdf extends LLDA {

    private int K;

    public LLDATfIdf(LLDACmdOption option) {
        super(option);
        K = option.K;
    }

    public LLDATfIdf(String metalabeler, String method, double beta, int iters,
            boolean parallel, String trainingFile, String test, int c, boolean inf, int t) {
        super(metalabeler, method, beta, iters, parallel, trainingFile, test, null, null, c, inf, t);
    }

    @Override
    public void train() {
        data = new DatasetTfIdf(this.trainingFile, false, false, 0, null, K);
        data.create(null);
        this.numFeatures = data.getV();
        this.M = data.getDocs().size();
        Model trnModel = null;
        if (parallel) {
            Model[] models = new Model[threads];
            for (int i = 0; i < threads; i++) {
                models[i] = new ModelTfIdf(data, i, beta, false, trainedModelName,
                        threads, iters, burnin);
            }
            //parallel Estimation
            pmc = new ParallelEstimation(models, threads);
            phi = pmc.startThreads();
        } else {
            TIntDoubleHashMap[] phiSum = null;
            for (int i = 0; i < chains; i++) {
                if ("cgs_p".equals(method)) {
                    System.out.println("CGS_p estimation");
                    trnModel = new CGS_pModelTfIdf(data, i, beta, trainedModelName, threads, iters, burnin);
                } else {
                    System.out.println("standard CGS estimation");
                    trnModel = new ModelTfIdf(data, i, beta, false, trainedModelName, threads, iters, burnin);
                }

                trnModel.estimate(true);
                //sum phi's
                if (i == 0) {
                    phiSum = trnModel.getPhi();
                } else {
                    for (int k = 0; k < trnModel.getK(); k++) {
                        TIntDoubleIterator it = trnModel.getPhi()[k].iterator();
                        while (it.hasNext()) {
                            it.advance();
                            phiSum[k].adjustOrPutValue(it.key(), it.value(), it.value());
                        }
                    }
                }
            }
            //average phi
            for (int k = 0; k < data.getK(); k++) {
                TIntDoubleIterator it = phiSum[k].iterator();
                while (it.hasNext()) {
                    it.advance();
                    it.setValue(it.value() / chains);
                }
            }

            System.out.println("Serial estimation finished");
//System.out.println(phiSum[0]);
            Model m = trnModel;
            m.setPhi(phiSum);
            m.save(15);
            phi = m.getPhi();

        }
        initMetalabeler();
    }

    @Override
    protected void initMetalabeler() {
        if (metalabelerFile != null) {
            System.out.println("Training metalabeler...");
            VectorizeLibSvm v = new VectorizeLibSvm();
            v.vectorizeTrain(this.trainingFile, this.trainingFile + "b");
            MetaModel ml = new MetaModel(numLabels, this.trainingFile + "b", null, numFeatures);
            ml.train(metaTrainLabels);
            ml.saveModel(metalabelerFile);
            metalabeler = ml.getModel();
        }
    }

    @Override
    public double[][] predictInternal(TIntHashSet mc) {
        TIntDoubleHashMap[] fi = Model.readPhi(trainedModelName + ".phi");
        this.numFeatures = Utils.max(fi);
        data = new DatasetTfIdf(testFile, true, true, numFeatures, fi, 0);
        data.create(null);
        M = data.getDocs().size();
        double[][] thetaSum = new double[M][data.getK()];
        Model newModel = null;
        System.out.println("Serial Inference");
        for (int i = 0; i < chains; i++) {
            newModel = new OnlyCGS_pPriorModelTfIdf(data, trainedModelName, threads, iters, burnin);
            newModel.inference();
            for (int m = 0; m < newModel.M; m++) {
                //sum up probabilities from the different markov chains
                for (int k = 0; k < newModel.K; k++) {
                    thetaSum[m][k] += newModel.getTheta()[m][k];
                }
            }

            if (i < chains - 1) {
                newModel = null;
                System.gc();
            }
        }
        //normalize
        System.out.println("Serial inference finished. Averaging....");
        for (int doc = 0; doc < thetaSum.length; doc++) {
            thetaSum[doc] = Utils.normalize(thetaSum[doc], 1.0);
        }
        newModel.setTheta(thetaSum);
        newModel.save(15);
        predictions = thetaSum;
        return predictions;
    }

    @Override
    protected void createDocMap() {

    }

    @Override
    public void createBipartitionsFromRanking(String metalabelerFile) {
        int corpusSize = predictions.length;
        double[] metalabelerPredictions = new double[corpusSize];
        VectorizeLibSvm v = new VectorizeLibSvm();
        v.vectorizeUnlabeled(this.testFile, this.testFilelibSVM);
        if (metalabelerFile != null) {

            metalabelerPredictions = MetaModel.getMetaModelPrediction(metalabelerFile,
                    numFeatures, corpusSize, this.testFilelibSVM, dictionary, this.corpus2);
        }
        for (int doc = 0; doc < corpusSize; doc++) {
            bipartitions.put(doc + "", new THashSet<>());
            //System.out.println(metalabelerPredictions[doc]);
            int d = 1 + (int) Utils.round(metalabelerPredictions[doc]);
            if (d < 1) {
                d = 1;
            }
            for (int k = 0; k < d; k++) {
                int label = Utils.maxIndex(predictions[doc]);
                predictions[doc][label] = Double.MIN_VALUE;
                bipartitions.get(doc + "").add(label + "");
            }
        }
    }

    public TreeMap<Integer, TObjectDoubleHashMap<String>> predictProbs2(TIntHashSet mc) {
        predictInternal(mc);
        int threshold = 40; //define how many labels to keep in the ranking, this is done for efficiency in storage
        TreeMap<Integer, TObjectDoubleHashMap<String>> probMap = new TreeMap<>();
        for (int doc = 0; doc < predictions.length; doc++) {

            if (!probMap.containsKey(doc)) {
                probMap.put(doc, new TObjectDoubleHashMap<>());
            }
            for (int k = 0; k < threshold; k++) {
                int l = Utils.maxIndex(predictions[doc]);
                probMap.get(doc).put(l + "", predictions[doc][l]);
                predictions[doc][l] = Double.MIN_VALUE;
            }
        }
        writeProbs2(probMap, "scores.txt");
        return probMap;
    }
    
        protected void writeProbs2(TreeMap<Integer, TObjectDoubleHashMap<String>> probMap, String scorestxt) {
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(scorestxt)))) {
            Iterator<Map.Entry<Integer, TObjectDoubleHashMap<String>>> it = probMap.entrySet().iterator();
            while (it.hasNext()) {
                Map.Entry<Integer, TObjectDoubleHashMap<String>> next = it.next();
                //System.out.println(next.getKey());
                TObjectDoubleIterator<String> it2 = next.getValue().iterator();
                TreeMap<Integer, Double> ordered  = new TreeMap<>();
                while(it2.hasNext()) {
                    it2.advance();
                    ordered.put(Integer.parseInt(it2.key()), it2.value());
                }
                StringBuilder sb = new StringBuilder();
                int i=0;
                Iterator<Map.Entry<Integer, Double>> it3 = ordered.entrySet().iterator();
                while(it3.hasNext()) {
                    Map.Entry<Integer, Double> n = it3.next();
                    sb.append(n.getKey()).append(":").append(Math.round(n.getValue() * 1000000.0) / 1000000.0);
                    if(i<ordered.size()-1) sb.append(" ");
                    i++;
                }
                sb.append("\n");
                writer.write(sb.toString()); 
            }
        } catch (Exception ex) {
            Logger.getLogger(MLClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}
