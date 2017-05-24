package gr.auth.csd.mlkd.mlclassification.labeledlda;

import gnu.trove.map.hash.TIntDoubleHashMap;

/**
 *
 * @author Yannis Papanikolaou
 * 29/03/14
 */
public abstract class ParallelMCMC {
//  Create parallel MCMC in order to speed up sampling for estimation/inference
    final int threads;

    public ParallelMCMC(int threads) {
        this.threads = threads;
    }   
    public abstract TIntDoubleHashMap[] startThreads();
}
