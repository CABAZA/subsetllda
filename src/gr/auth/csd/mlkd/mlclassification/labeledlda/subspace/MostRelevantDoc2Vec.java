package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gr.auth.csd.mlkd.atypon.CmdOption;
import gr.auth.csd.mlkd.atypon.mlclassification.deeplearning.ParagraphVectorModel;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizewithDoc2Vec;
import java.io.File;
import java.util.ArrayList;
import java.util.Set;

/**
 *
 * @author Yannis Papanikolaou
 *
 * Retrieves the n-most relevaqnt documents and creates a subspace of the labels
 * using unsupervised LDA set to be used for LLDA inference
 */
public class MostRelevantDoc2Vec extends MostRelevantTfIdf {

    private final ParagraphVectorModel pvm;

    public MostRelevantDoc2Vec(int n, CmdOption op) {
        super(n, op);
        File trainedPV = new File("PV.model");
        if (trainedPV.exists()) {
            pvm = ParagraphVectorModel.read("PV.model");
        } else {
            CorpusJSON corpus = new CorpusJSON(op.trainingFile);
            pvm = new ParagraphVectorModel(corpus, 200);
            pvm.train();
            pvm.save("PV.model");
        }
    }

    @Override
    public ArrayList<TObjectDoubleHashMap<String>> mostRelevant() {
        System.out.println("Most relevant labels...");
        VectorizewithDoc2Vec vectorize = new VectorizewithDoc2Vec(dic, ls, new CorpusJSON(trainFile));
        CorpusJSON training = new CorpusJSON(trainFile);
        CorpusJSON test = new CorpusJSON(testFile);
        trainingFileLabels = gettrainingFileLabels(training);

        trainVectors = vectorize.vectorizeUnlabeled(training);
        testVectors = vectorize.vectorizeUnlabeled(test);
        
        
        //System.out.println("theta_p " + Arrays.toString(testTheta_p[0]));
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
}
