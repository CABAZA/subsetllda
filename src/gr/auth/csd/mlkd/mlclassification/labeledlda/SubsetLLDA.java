package gr.auth.csd.mlkd.mlclassification.labeledlda;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.utils.LLDACmdOption;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.Model;
import gr.auth.csd.mlkd.mlclassification.labeledlda.subspace.MostRelevantJaccard;
import gr.auth.csd.mlkd.mlclassification.labeledlda.subspace.MostRelevantTfIdf;
import gr.auth.csd.mlkd.mlclassification.labeledlda.subspace.SubsetModel;
import gr.auth.csd.mlkd.utils.Utils;

import java.io.File;

public class SubsetLLDA extends LLDA {

    private final String possibleLabels;
    private MostRelevantTfIdf mr;
    private final String method = "tf-idf";//"all";

    public SubsetLLDA(LLDACmdOption option) {
        super(option);
        this.possibleLabels = option.testFile+".alpha";
    }

    @Override
    public void train() {
        super.train();
        //mr = new MostRelevantTfIdf(10, this.testFile, this.trainingFile);
        //mr = new MostRelevantJaccard(10, this.testFile, this.trainingFile);
    }

    @Override
    public void predictInternal() {
        TIntDoubleHashMap[] fi = Model.readPhi(trainedModelName + ".phi");
        this.numFeatures = Utils.max(fi);
        if (!new File(testFile + ".wlabels").exists()) {
            mr = new MostRelevantTfIdf(10, this.testFile, this.trainingFile);
            //mr = new MostRelevantJaccard(10, this.testFile, this.trainingFile);
            mr.mostRelevant();
        }
        data = new DatasetTfIdf(testFile + ".wlabels", true, numFeatures, fi);
        data.create(false);
        System.out.println("Serial Inference");
        SubsetModel newModel = new SubsetModel(data, trainedModelName, threads, iters, burnin, possibleLabels);
        newModel.inference();
        //normalize
        System.out.println("Serial inference finished.");
        newModel.save(15);
        predictions = newModel.getTheta();
    }

}
