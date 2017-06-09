package gr.auth.csd.mlkd.lda;

import gr.auth.csd.mlkd.atypon.LDACmdOption;
import gr.auth.csd.mlkd.atypon.lda.DatasetTfIdf;
import gr.auth.csd.mlkd.lda.models.InferenceModel;
import gr.auth.csd.mlkd.lda.models.Model;
import gr.auth.csd.mlkd.mlclassification.labeledlda.Dataset;

public class LDA {

    static Dataset data;

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
        this.method = option.method;
        trainedPhi = option.modelName + ".phi";
        data = new DatasetTfIdf(false, option.inf, option.K, trainedPhi);
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
        Model trnModel;
        data.create(true);
        trnModel = new Model(data, a, false, b, perp, niters, nburnin, modelName, samplingLag);
        return trnModel.estimate(true);
    }

    public Model inference() {
        Model newModel;
        data.create(true);
        newModel = new InferenceModel(data, a, perp, niters, nburnin, modelName, samplingLag);
        newModel.initialize();
        newModel.inference();
        return newModel;
    }
}
