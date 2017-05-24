package gr.auth.csd.mlkd.mlclassification.labeledlda.subspace;

import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.utils.CmdOption;
import gr.auth.csd.mlkd.utils.Utils;


/**
 *
 * @author Yannis Papanikolaou
 *
 * Retrieves the n-most relevaqnt documents and creates a subspace of the labels
 * set to be used for LLDA inference
 */
public class MostRelevantLibSvmJaccard extends MostRelevantLibSvm {

    public MostRelevantLibSvmJaccard(int n, CmdOption option) {
        super(n, option);
    }

    public MostRelevantLibSvmJaccard(int n, String testFile, String trainFile) {
        super(n, testFile, trainFile);
    }

    @Override
    public double similarity(TIntDoubleHashMap get, TIntDoubleHashMap get0) {
        return Utils.jaccard(get, get0);
    }
    
    

    

}
