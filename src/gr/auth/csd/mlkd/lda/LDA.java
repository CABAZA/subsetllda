package gr.auth.csd.mlkd.atypon.lda;

import gr.auth.csd.mlkd.atypon.LDACmdOption;
import gr.auth.csd.mlkd.atypon.lda.models.CVB0InferenceModel;
import gr.auth.csd.mlkd.atypon.lda.models.CVB0Model;
import gr.auth.csd.mlkd.atypon.lda.models.EstimationConvergenceExperimentModel;
import gr.auth.csd.mlkd.atypon.lda.models.InferenceExpModel;
import gr.auth.csd.mlkd.atypon.lda.models.InferenceModel;
import gr.auth.csd.mlkd.atypon.lda.models.Model;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import java.io.File;

public class LDA {

    static DatasetBoW data;
    CorpusJSON corpus;
    Dictionary dictionary;

    final String method;
    final String testFile;
    final double a, b;
    final boolean perp;
    final int niters, nburnin;
    final String modelName;
    protected final int chains;
    protected final int samplingLag;
    final String trainedPhi;

    public LDA(LDACmdOption option) {
        corpus = new CorpusJSON(option.trainingFile);
        if (!(new File(option.dictionary)).exists()) {
            dictionary = Dictionary.readDictionary(option.dictionary);
        } else {
            dictionary = new Dictionary(corpus, option.lowUnigrams, option.highUnigrams, option.lowBigrams, option.highBigrams);
        }
        this.method = option.method;
        trainedPhi = option.modelName + ".phi";
        data = new DatasetBoW(dictionary, false, option.inf, option.K, trainedPhi);
        this.chains = option.chains;
        this.modelName = option.modelName;
        this.niters = option.niters;
        nburnin = option.nburnin;
        this.a = option.alpha;
        this.b = option.beta;
        this.perp = option.perplexity;
        this.testFile = option.testFile;
        this.samplingLag = option.samplingLag;
    }

    public double[][] estimation() {
        Model trnModel = null;
        data.create(corpus);
        if (null != method) {
            switch (method) {
                case "conv":
                    trnModel = new EstimationConvergenceExperimentModel(data, a, false, b, perp, niters, nburnin, modelName, samplingLag);
                    break;
                case "cvb0":
                    trnModel = new CVB0Model(data, a, false, b, perp, niters, nburnin, modelName, samplingLag);
                    break;
                default:
                    trnModel = new Model(data, a, false, b, perp, niters, nburnin, modelName, samplingLag);
                    break;
            }
        }
        return trnModel.estimate(true);
    }

    public Model inference() {
        Model newModel = null;
        data.create(new CorpusJSON(testFile));
        if (method.equals("cvb0")) {
            newModel = new CVB0InferenceModel(data, a, perp, niters, nburnin, modelName, samplingLag);
        } else {
            newModel = new InferenceModel(data, a, perp, niters, nburnin, modelName, samplingLag);
        }
        newModel.initialize();
        newModel.inference();
        return newModel;
    }

    public InferenceExpModel inference2() {
        InferenceExpModel newModel = null;
        data.create(new CorpusJSON(testFile));
        newModel = new InferenceExpModel(data, a, perp, niters, nburnin, modelName, samplingLag);
        newModel.initialize();
        newModel.inference();
        return newModel;
    }
}
