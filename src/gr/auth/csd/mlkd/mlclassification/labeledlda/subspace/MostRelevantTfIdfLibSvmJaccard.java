package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.atypon.CmdOption;
import gr.auth.csd.mlkd.atypon.utils.Utils;

/**
 *
 * @author Yannis Papanikolaou
 *
 * Retrieves the n-most relevaqnt documents and creates a subspace of the labels
 * set to be used for LLDA inference
 */
public class MostRelevantTfIdfLibSvmJaccard extends MostRelevantTfIdfLibSvm {

    public MostRelevantTfIdfLibSvmJaccard(int n, CmdOption option) {
        super(n, option);
    }

    public MostRelevantTfIdfLibSvmJaccard(int n, String testFile, String trainFile) {
        super(n, testFile, trainFile);
    }

    @Override
    public double similarity(TIntDoubleHashMap get, TIntDoubleHashMap get0) {
        return Utils.jaccard(get, get0);
    }
    
    

    

}
