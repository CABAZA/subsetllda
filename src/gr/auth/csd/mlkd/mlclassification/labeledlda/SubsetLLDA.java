package gr.auth.csd.mlkd.mlclassification.labeledlda;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.set.hash.TIntHashSet;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.Model;
import gr.auth.csd.mlkd.mlclassification.labeledlda.subspace.MostRelevantLibSvm;
import gr.auth.csd.mlkd.mlclassification.labeledlda.subspace.SubsetModel;
import gr.auth.csd.mlkd.utils.LLDACmdOption;
import gr.auth.csd.mlkd.utils.Utils;

import java.io.File;

public class SubsetLLDA extends LLDA {

    private final String possibleLabels;
    private MostRelevantLibSvm mr;
    private String method = "tf-idf";//"all";

    public SubsetLLDA(LLDACmdOption option) {
        super(option);
        this.possibleLabels = "alpha";
    }

    @Override
    public void train() {
        super.train();
        mr = new MostRelevantLibSvm(10, this.testFile, this.trainingFile);
        //mr = new MostRelevantTfIdfLibSvmJaccard(10, this.testFile, this.trainingFile);
    }

    @Override
    public double[][] predictInternal(TIntHashSet mc) {
        TIntDoubleHashMap[] fi = Model.readPhi(trainedModelName + ".phi");
        this.numFeatures = Utils.max(fi);
        if (!new File(testFile + ".wlabels").exists()) {
            mr = new MostRelevantLibSvm(10, this.testFile, this.trainingFile);
            //mr = new MostRelevantTfIdfLibSvmJaccard(10, this.testFile, this.trainingFile);
            mr.mostRelevant();
        }
        data = new DatasetTfIdf(testFile + ".wlabels", false, true, numFeatures, fi, 0);
        data.create();
        System.out.println("Serial Inference");
        SubsetModel newModel = new SubsetModel(data, trainedModelName, threads, iters, burnin, possibleLabels);
        newModel.inference();
        //normalize
        System.out.println("Serial inference finished.");
        newModel.save(15);
        predictions = newModel.getTheta();
        return predictions;
    }

}
