package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gr.auth.csd.mlkd.atypon.CmdOption;
import gr.auth.csd.mlkd.atypon.LDACmdOption;
import gr.auth.csd.mlkd.atypon.lda.LDACGS_p;
import gr.auth.csd.mlkd.atypon.lda.models.InferenceModel;
import gr.auth.csd.mlkd.atypon.lda.models.Model;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import java.io.File;
import java.util.ArrayList;
import java.util.Set;
import gr.auth.csd.mlkd.atypon.utils.Utils;

/**
 *
 * @author Yannis Papanikolaou
 *
 * Retrieves the n-most relevaqnt documents and creates a subspace of the labels
 * using unsupervised LDA set to be used for LLDA inference
 */
public class MostRelevantLDA extends MostRelevantTfIdf {

    private final LDACmdOption option;
    private LDACGS_p lda;

    public MostRelevantLDA(int n, CmdOption op) {
        super(n, op);
        option = new LDACmdOption();
        option.trainingFile = op.trainingFile;
        option.testFile = op.testFile;
        option.K = 200;
        lda = new LDACGS_p(option);
        lda.estimation();
        double[][] trainedTheta_p = lda.getTrnModel().getTheta_p();
        //System.out.println(Arrays.toString(trainedTheta_p[0]));
        trainVectors = transform(trainedTheta_p);
        File file = new File(option.modelName + ".phi");
        File file3 = new File(option.modelName + ".phi_p");
        boolean success = file3.renameTo(file);
    }

    public MostRelevantLDA(int n, CmdOption op, String trainedTheta) {
        super(n, op);
        option = new LDACmdOption();
        option.trainingFile = op.trainingFile;
        option.testFile = op.testFile;
        option.K = 200;
        double[][] trainedTheta_p = InferenceModel.readTheta_p(trainedTheta);
        trainVectors = transform(trainedTheta_p);
        //System.out.println(Arrays.toString(trainedTheta_p[0]));
    }

    @Override
    public ArrayList<TObjectDoubleHashMap<String>> mostRelevant() {
        System.out.println("Most relevant labels...");

        if (lda == null) {
            File file = new File(option.modelName + ".phi");
            File file3 = new File(option.modelName + ".phi_p");
            boolean success = file3.renameTo(file);
            lda = new LDACGS_p(option);
        }
        Model model = lda.inference();
        double[][] testTheta_p = ((InferenceModel) model).getTheta_p();
        //System.out.println("theta_p " + Arrays.toString(testTheta_p[0]));
        trainingFileLabels = gettrainingFileLabels(new CorpusJSON(option.trainingFile));

        testVectors = transform(testTheta_p);
        return findMostRelevant();
    }

    private ArrayList<TIntDoubleHashMap> transform(double[][] trainedTheta_p) {
        ArrayList<TIntDoubleHashMap> transformed = new ArrayList<>();
        for (int i = 0; i < trainedTheta_p.length; i++) {
            transformed.add(i, new TIntDoubleHashMap());
            for (int j = 0; j < trainedTheta_p[0].length; j++) {
                if (trainedTheta_p[i][j] != 0) {
                    transformed.get(i).put(j, trainedTheta_p[i][j]);
                }
            }
        }
        return transformed;
    }

    @Override
    public double similarity(TIntDoubleHashMap get, TIntDoubleHashMap get0) {
        //return super.similarity(get, get0);
        return Utils.hellingerDistance(get, get0);
    }
    
}
