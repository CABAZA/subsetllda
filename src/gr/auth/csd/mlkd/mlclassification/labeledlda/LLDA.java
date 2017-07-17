package gr.auth.csd.mlkd.mlclassification.labeledlda;

import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.mlclassification.MLClassifier;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.EstimationCGSpModel;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.InferenceCGSpModel;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.Model;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.ModelTfIdf;
import gr.auth.csd.mlkd.utils.LLDACmdOption;
import gr.auth.csd.mlkd.utils.Utils;

import java.util.ArrayList;

public class LLDA extends MLClassifier {

    private int K;
    Dataset data;
    ParallelMCMC pmc;
    protected int M;
    protected String method;
    TIntDoubleHashMap[] phi;
    final double beta;
    protected int iters;
    int burnin = 50;
    final String trainedModelName;
    protected final boolean parallel;
    protected int chains = 1;
    int numFeatures;

    public LLDA(LLDACmdOption option) {
        super(option.trainingFile, option.testFile, option.threads);
        this.method = option.method;
        this.parallel = option.parallel;
        this.beta = option.beta;
        this.burnin = option.nburnin;
        this.iters = option.niters;
        this.trainedModelName = option.modelName;
        this.chains = option.chains;
    }

    @Override
    public void train() {
        data = new DatasetTfIdf(this.trainingFile, false, 0, null);
        data.create(true);
        this.numFeatures = data.getV();
        this.M = data.getDocs().size();
        Model trnModel = null;
        if (parallel) {
            Model[] models = new Model[threads];
            for (int i = 0; i < threads; i++) {
                models[i] = new ModelTfIdf(data, i, beta, false, trainedModelName,
                        threads, iters, burnin);
            }
            //parallel Estimation
            pmc = new ParallelEstimation(models, threads);
            phi = pmc.startThreads();
        } else {
            TIntDoubleHashMap[] phiSum = null;
            for (int i = 0; i < chains; i++) {
                if ("cgs_p".equals(method)) {
                    System.out.println("CGS_p estimation");
                    trnModel = new EstimationCGSpModel(data, i, beta, trainedModelName, threads, iters, burnin);
                } else {
                    System.out.println("standard CGS estimation");
                    trnModel = new ModelTfIdf(data, i, beta, false, trainedModelName, threads, iters, burnin);
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
    }

    @Override
    public void predictInternal() {
        TIntDoubleHashMap[] fi = Model.readPhi(trainedModelName + ".phi");
        this.numFeatures = Utils.max(fi);
        data = new DatasetTfIdf(testFile, true, numFeatures, fi);
        data.create(true);
        M = data.getDocs().size();
//        ArrayList<TIntDoubleHashMap> thetaSum = new ArrayList<>();
//        for (int m = 0; m < M; m++) {
//            thetaSum.add(new TIntDoubleHashMap());
//        }
        Model newModel = null;
        System.out.println("Serial Inference");
        //for (int i = 0; i < chains; i++) {
            newModel = new InferenceCGSpModel(data, trainedModelName, threads, iters, burnin);
            newModel.inference();
/*            for (int m = 0; m < M; m++) {
                //sum up probabilities from the different markov chains
                for (int k = 0; k < newModel.K; k++) {
                    double th = newModel.getTheta().get(m).get(k);
                    thetaSum.get(m).adjustOrPutValue(k, th,th);
                }
            }
*/            /*if (i < chains - 1) {
                newModel = null;
                System.gc();
            }
        //}
        //normalize
        System.out.println("Serial inference finished. Averaging....");
//        for (int doc = 0; doc < thetaSum.size(); doc++) {
//            thetaSum.set(doc, Utils.normalize(thetaSum.get(doc), 1.0));
//        }
        newModel.setTheta(thetaSum);
        newModel.save(15);
        */predictions = newModel.getTheta();//thetaSum;
    }
}
