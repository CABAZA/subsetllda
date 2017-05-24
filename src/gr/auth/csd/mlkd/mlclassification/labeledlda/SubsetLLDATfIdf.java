package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.set.hash.TIntHashSet;
import gr.auth.csd.mlkd.atypon.LLDACmdOption;
import gr.auth.csd.mlkd.atypon.lda.DatasetTfIdf;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.models.Model;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace.MostRelevantTfIdfLibSvm;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace.MostRelevantTfIdfLibSvmJaccard;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace.SubsetModelTfIdf;
import gr.auth.csd.mlkd.atypon.utils.Utils;
import java.io.File;

public class SubsetLLDATfIdf extends LLDATfIdf {

    private final String possibleLabels;
    private MostRelevantTfIdfLibSvm mr;
    private String method = "tf-idf";//"all";

    public SubsetLLDATfIdf(LLDACmdOption option) {
        super(option);
        this.possibleLabels = "alpha";
    }

    public SubsetLLDATfIdf(String metalabeler, String method, double beta, int iters,
            boolean parallel, String trainingFile, String test, int c, boolean inf, int t) {
        super(metalabeler, method, beta, iters, parallel, trainingFile, test, c, inf, t);
        this.possibleLabels = "alpha";
    }

    @Override
    public void train() {
        super.train();
        mr = new MostRelevantTfIdfLibSvm(10, this.testFile, this.trainingFile);
        //mr = new MostRelevantTfIdfLibSvmJaccard(10, this.testFile, this.trainingFile);
    }

    @Override
    public double[][] predictInternal(TIntHashSet mc) {
        TIntDoubleHashMap[] fi = Model.readPhi(trainedModelName + ".phi");
        this.numFeatures = Utils.max(fi);
        if (!new File(testFile + ".wlabels").exists()) {
            mr = new MostRelevantTfIdfLibSvm(10, this.testFile, this.trainingFile);
            //mr = new MostRelevantTfIdfLibSvmJaccard(10, this.testFile, this.trainingFile);
            mr.mostRelevant();
        }
        data = new DatasetTfIdf(testFile + ".wlabels", false, true, numFeatures, fi, 0);
        data.create(null);
        System.out.println("Serial Inference");
        SubsetModelTfIdf newModel = new SubsetModelTfIdf(data, trainedModelName, threads, iters, burnin, possibleLabels);
        newModel.inference();
        //normalize
        System.out.println("Serial inference finished.");
        newModel.save(15);
        predictions = newModel.getTheta();
        return predictions;
    }

}
