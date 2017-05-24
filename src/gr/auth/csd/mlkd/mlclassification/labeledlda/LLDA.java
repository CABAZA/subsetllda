package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda;

import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.set.hash.TIntHashSet;
import gr.auth.csd.mlkd.atypon.LLDACmdOption;
import gr.auth.csd.mlkd.atypon.lda.Dataset;
import gr.auth.csd.mlkd.atypon.lda.DatasetBoW;

import gr.auth.csd.mlkd.atypon.mlclassification.MLClassifier;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.CGS_pModel;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.CVB0InferenceModel;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.CVB0Model;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.Model;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.OnlyCGS_pPriorModel;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.PriorModel;
import gr.auth.csd.mlkd.atypon.mlclassification.svm.MetaModel;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizeJSON;
import gr.auth.csd.mlkd.atypon.utils.Utils;

public class LLDA extends MLClassifier {

    Dataset data;
    ParallelMCMC pmc;
    String metalabelerFile;
    protected int M;
    protected String method;
    TIntDoubleHashMap[] phi;
    de.bwaldvogel.liblinear.Model metalabeler;
    final double beta;
    protected int iters;
    int burnin = 50;
    final String trainedModelName;
    protected final boolean parallel;
    protected int chains = 1;
    final String metaTrainLabels = "metaTrainLabels";
    private final String trainLibsvm = "train.Libsvm";

    public LLDA(LLDACmdOption option) {
        super(option.trainingFile, option.testFile, option.dictionary, option.labels, option.threads);
        this.metalabelerFile = option.metalabelerFile;
        this.method = option.method;
        this.parallel = option.parallel;
        this.beta = option.beta;
        this.burnin = option.nburnin;
        this.iters = option.niters;
        this.trainedModelName = option.modelName;
        this.chains = option.chains;
    }

    public LLDA(String metalabeler, String method, double beta, int iters, boolean parallel,
            String trainingFile, String test, Dictionary dic, Labels labels, int c, boolean inf, int t) {
        super(trainingFile, test, dic, labels, t);
        this.metalabelerFile = metalabeler;
        this.method = method;
        this.beta = beta;
        this.iters = iters;
        this.trainedModelName = trainingFile + ".model";
        this.parallel = parallel;
        this.chains = c;
    }

    public LLDA(String metalabeler, String method, double beta, int iters, boolean parallel,
            Dictionary dictionary, Labels labels, CorpusJSON trainingCorpus,
            CorpusJSON testCorpus, boolean inf, int c, String trainingFile, int t) {
        super(dictionary, labels, trainingCorpus, testCorpus, t);
        this.metalabelerFile = metalabeler;
        this.method = method;
        this.beta = beta;
        this.iters = iters;
        this.trainedModelName = trainingFile + ".model";
        this.parallel = parallel;
        this.chains = c;
    }

    public static void main(String args[]) {
        LLDACmdOption option = new LLDACmdOption(args);
        LLDA llda = new LLDA(option);
        llda.train();
        llda.predict(null);
    }

    @Override
    public void train() {
        data = new DatasetBoW(dictionary, globalLabels, false, false);
        data.create((CorpusJSON) corpus);
        this.M = data.getDocs().size();
        Model trnModel = null;
        if (parallel) {
            Model[] models = new Model[threads];
            for (int i = 0; i < threads; i++) {
                if ("cvb0".equals(method)) {
                    System.out.println("CVB0 estimation");
                    models[i] = new CVB0Model(data, i, beta, false, trainedModelName, threads, iters, burnin);
                } else if ("cgs_p".equals(method)) {
                    System.out.println("CGS_p estimation");
                    models[i] = new CGS_pModel(data, i, beta, trainedModelName, threads, iters, burnin);

                } else {
                    System.out.println("standard CGS estimation");
                    models[i] = new Model(data, i, beta, false, trainedModelName, threads, iters, burnin);
                }
            }
            //parallel Estimation
            pmc = new ParallelEstimation(models, threads);
            phi = pmc.startThreads();
        } else {
            TIntDoubleHashMap[] phiSum = null;
            for (int i = 0; i < chains; i++) {

                if ("cvb0".equals(method)) {
                    System.out.println("CVB0 estimation");
                    trnModel = new CVB0Model(data, i, beta, false, trainedModelName, threads, iters, burnin);
                } else if ("cgs_p".equals(method)) {
                    System.out.println("CGS_p estimation");
                    trnModel = new CGS_pModel(data, i, beta, trainedModelName, threads, iters, burnin);
                } else {
                    System.out.println("standard CGS estimation");
                    trnModel = new Model(data, i, beta, false, trainedModelName, threads, iters, burnin);
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
    public double[][] predictInternal(TIntHashSet mc) {
        data = new DatasetBoW(dictionary, globalLabels, true, true);
        data.create((CorpusJSON) corpus2);
        M = data.getDocs().size();
        double[][] thetaSum = new double[M][data.getK()];
        Model newModel = null;
        System.out.println("Serial Inference");
        for (int i = 0; i < chains; i++) {
            newModel = createModel();
            newModel.inference();
            for (int m = 0; m < newModel.M; m++) {
                //sum up probabilities from the different markov chains
                for (int k = 0; k < newModel.K; k++) {
                    thetaSum[m][k] += newModel.getTheta()[m][k];
                }
            }
//            for (int k = 0; k < newModel.K; k++) {
//                System.out.print(globalLabels.getLabel(k + 1) + "=" + thetaSum[0][k] + " ");
//            }
//            System.out.println();

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
    public void createBipartitions() {
        createBipartitionsFromRanking(metalabelerFile);
    }

    protected Model createModel() {
        Model newModel;
        switch (method) {
            case "cvb0":
                System.out.println("CVB0 Inference");
                newModel = new CVB0InferenceModel((DatasetBoW)data, trainedModelName, threads, iters, burnin);
                break;
            case "cgs_p":
                System.out.println("CGS_p Inference");
                newModel = new OnlyCGS_pPriorModel((DatasetBoW)data, trainedModelName, threads, iters, burnin);
                break;
            default:
                System.out.println("Standard Inference");
                newModel = new PriorModel((DatasetBoW)data, trainedModelName, threads, iters, burnin);
                break;
        }
        return newModel;
    }

    protected void initMetalabeler() {
        if (metalabelerFile != null) {
            System.out.println("Training metalabeler...");
            VectorizeJSON vectorize = new VectorizeJSON(dictionary, true, globalLabels);
            vectorize.vectorizeTrain(corpus, "train.Libsvm", "trainLabels", "metaTrainLabels");
            MetaModel ml = new MetaModel(numLabels, trainLibsvm, null, dictionary.getId().size());
            ml.train(metaTrainLabels);
            ml.saveModel(metalabelerFile);
            metalabeler = ml.getModel();
        }
    }

    public double[][] getPredictions() {
        return predictions;
    }

    public void setPredictions(double[][] predictions) {
        this.predictions = predictions;
    }

    public TIntDoubleHashMap[] getPhi() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
