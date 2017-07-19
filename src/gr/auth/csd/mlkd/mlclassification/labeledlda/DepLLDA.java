package gr.auth.csd.mlkd.mlclassification.labeledlda;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.utils.LLDACmdOption;
import gr.auth.csd.mlkd.lda.LDASerial;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.DependencyModel;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.InferenceCGSpModel;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.Model;
import gr.auth.csd.mlkd.utils.Utils;
import java.util.ArrayList;

public class DepLLDA extends LLDA {

    private final String trainedPhi2;
    private final double g;
    private final LLDACmdOption option;

    public DepLLDA(LLDACmdOption option) {
        super((LLDACmdOption) option);
        this.option = option;
        g = ((LLDACmdOption) option).gamma;
        trainedPhi2 = option.modelName + ".phi2";
    }

    @Override
    public void train() {
        System.out.println("Running unsupervised LDA on labels...");
        LDASerial ldaserial = new LDASerial(option);
        ldaserial.estimation2();
        ldaserial.writePhi(option.modelName + ".phi2");
        System.out.println("Finished unsupervised LDA on labels.");
        super.train();
    }

    @Override
    public void predictInternal() {
        TIntDoubleHashMap[] fi = Model.readPhi(trainedModelName + ".phi");
        this.numFeatures = Utils.max(fi)+1;
        data = new DatasetTfIdf(testFile, true, numFeatures, fi);
        data.create(true);
        M = data.getDocs().size();
        ArrayList<TIntDoubleHashMap> thetaSum = new ArrayList<>();
        for (int m = 0; m < M; m++) {
            thetaSum.add(new TIntDoubleHashMap());
        }
        Model newModel = null;
        System.out.println("Serial Inference");

        double[][][] phiArray = LDASerial.readPhi2(trainedPhi2);
        int numOfLDAModels = phiArray.length;

        for (int i = 0; i < chains / numOfLDAModels; i++) {
            for (int j = 0; j < numOfLDAModels; j++) {
                newModel = new DependencyModel(data, iters, burnin, trainedModelName, phiArray[0], g, threads);
                newModel.inference();
                if (i + j == 0) {
                    thetaSum = newModel.getTheta();
                } else {
                    //predictions = newModel.getTheta();

                    for (int m = 0; m < newModel.M; m++) {
                        //sum up probabilities from the different markov chains
                        for (int k = 0; k < newModel.K; k++) {
                            double val = newModel.getTheta().get(m).get(k);
                            thetaSum.get(m).adjustOrPutValue(k, val, val);
                        }
                    }
                }
                if (i < chains - 1) {
                    newModel = null;
                    System.gc();
                }
            }
        }
        //normalize
        System.out.println("Serial inference finished. Averaging....");
        for (int doc = 0; doc < thetaSum.size(); doc++) {
            thetaSum.set(doc, Utils.normalize(thetaSum.get(doc), 1));
        }
        newModel.setTheta(thetaSum);
        newModel.save(15);
        predictions = thetaSum;
    }
}
