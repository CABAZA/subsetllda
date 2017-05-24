package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda;

import gnu.trove.set.hash.TIntHashSet;
import gr.auth.csd.mlkd.atypon.LLDACmdOption;
import gr.auth.csd.mlkd.atypon.lda.DatasetBoW;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.Model;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace.MostRelevantAll;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace.MostRelevantDoc2Vec;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace.MostRelevantLDA;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace.MostRelevantTfIdf;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace.SubsetModel;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.utils.Utils;
import java.io.File;

public class SubsetLLDA extends LLDA {

    public static void main(String args[]) {
        LLDACmdOption option = new LLDACmdOption(args);
        SubsetLLDA llda = new SubsetLLDA(option);
        llda.train();
        llda.predict(null);
    }
    private final String possibleLabels;
    private LLDACmdOption option;
    private MostRelevantTfIdf mr;
    private String method = "tf-idf";//"all";

    public SubsetLLDA(LLDACmdOption option) {
        super(option);
        this.option = option;
        this.possibleLabels = option.possibleLabels;
    }

    public SubsetLLDA(String metalabeler, String method, double beta, int iters, boolean parallel, String trainingFile, String test, Dictionary dic, Labels labels, int c, boolean inf, int t) {
        super(metalabeler, method, beta, iters, parallel, trainingFile, test, dic, labels, c, inf, t);
        this.possibleLabels = "alpha"+test;
    }

    public SubsetLLDA(String metalabeler, String method, double beta, int iters, boolean parallel, Dictionary dictionary, Labels labels, CorpusJSON trainingCorpus, CorpusJSON testCorpus, boolean inf, int c, String trainingFile, int t) {
        super(metalabeler, method, beta, iters, parallel, dictionary, labels, trainingCorpus, testCorpus, inf, c, trainingFile, t);
        this.possibleLabels = "alpha";
    }

    @Override
    public void train() {
        super.train();
        mr = ("tf-idf".equals(method))?new MostRelevantTfIdf(10, this.testFile, this.trainingFile,
                dictionary, globalLabels):("lda".equals(method))
                ?new MostRelevantLDA(10, option):("d2v".equals(method))
                ?new MostRelevantDoc2Vec(10, option):new MostRelevantAll(10, option);

    }

    @Override
    public double[][] predictInternal(TIntHashSet mc) {
        if(!new File(testFile + ".wlabels").exists()) {
            mr = ("tf-idf".equals(method))?new MostRelevantTfIdf(10, this.testFile, this.trainingFile,
                dictionary, globalLabels):("lda".equals(method))
                ?new MostRelevantLDA(10, option):("d2v".equals(method))
                ?new MostRelevantDoc2Vec(10, option):new MostRelevantAll(10, option);
            mr.mostRelevant();
        }
        corpus2 = new CorpusJSON(testFile + ".wlabels");
        data = new DatasetBoW(dictionary, globalLabels, false, true);
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
            if (i < chains - 1) {
                newModel = null;
                System.gc();
            }
            //System.out.println(Arrays.toString(thetaSum[0]));
        }
        //normalize
        System.out.println("Serial inference finished. Averaging....");
        for (int doc = 0; doc < thetaSum.length; doc++) {
            thetaSum[doc] = Utils.normalize(thetaSum[doc], 1);
        }
        newModel.setTheta(thetaSum);
        newModel.save(15);
        predictions = thetaSum;
        return predictions;
    }

    @Override
    protected Model createModel() {
        System.out.println("Subset LLDA Inference");
        Model newModel = new SubsetModel(data, trainedModelName, threads, iters, burnin, possibleLabels);
        return newModel;
    }
}
