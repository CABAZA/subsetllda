package gr.auth.csd.mlkd.atypon.lda;

import gr.auth.csd.mlkd.atypon.LDACmdOption;
import gr.auth.csd.mlkd.atypon.lda.models.ConvergenceExperiment3Model;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;




public class LDAConvExp extends LDA {

    public LDAConvExp(LDACmdOption option) {
        super(option);
    }

    public ConvergenceExperiment3Model inference3() {
        ConvergenceExperiment3Model newModel = null;
        data.create(new CorpusJSON(testFile));
        newModel = new ConvergenceExperiment3Model(data, a, perp, niters, nburnin, modelName, samplingLag);
        newModel.initialize();
        newModel.inference();
        return newModel;
    }
}
