package gr.auth.csd.mlkd.mlclassification.labeledlda;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.InferenceCGSpModel;
import gr.auth.csd.mlkd.utils.LLDACmdOption;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.Model;
import gr.auth.csd.mlkd.mlclassification.labeledlda.subspace.MostRelevantJaccard;
import gr.auth.csd.mlkd.mlclassification.labeledlda.subspace.MostRelevantTfIdf;
import gr.auth.csd.mlkd.mlclassification.labeledlda.subspace.SubsetModel;
import gr.auth.csd.mlkd.utils.Utils;

import java.io.File;
import java.util.ArrayList;

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
        this.numFeatures = Utils.max(fi)+1;
        if (!new File(testFile + ".wlabels").exists()) {
            mr = new MostRelevantTfIdf(5, this.testFile, this.trainingFile);
            //mr = new MostRelevantJaccard(10, this.testFile, this.trainingFile);
            mr.mostRelevant();
        }
        data = new DatasetTfIdf(testFile + ".wlabels", true, numFeatures, fi);
        data.create(false);
        M = data.getDocs().size();
        System.out.println("Subset LLDA - Serial Inference");
                ArrayList<TIntDoubleHashMap> thetaSum = new ArrayList<>();
        for (int m = 0; m < M; m++) {
            thetaSum.add(new TIntDoubleHashMap());
        }
        Model newModel = null;
        for (int i = 0; i < chains; i++) {
            newModel = new SubsetModel(data, trainedModelName, threads, iters, burnin, possibleLabels);
            newModel.inference();
            for (int m = 0; m < M; m++) {
                //sum up probabilities from the different markov chains
                for (int k = 0; k < newModel.K; k++) {
                    double th = newModel.getTheta().get(m).get(k);
                    thetaSum.get(m).adjustOrPutValue(k, th,th);
                }
            }
            if (i < chains - 1) {
                newModel = null;
                System.gc();
            }
        }
        //normalize
        System.out.println("Serial inference finished. Averaging....");
        for (int doc = 0; doc < thetaSum.size(); doc++) {
            thetaSum.set(doc, Utils.normalize(thetaSum.get(doc), 1.0));
        }
        newModel.setTheta(thetaSum);
        newModel.save(15);
        predictions = thetaSum;
    }

}
