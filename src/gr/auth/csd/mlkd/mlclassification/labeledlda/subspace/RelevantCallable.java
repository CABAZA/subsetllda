/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace;

import gnu.trove.map.hash.TObjectDoubleHashMap;
import java.util.concurrent.Callable;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
class RelevantCallable<T> implements Callable<TObjectDoubleHashMap<String>> {
    int i;
    private final MostRelevant mr;

    public RelevantCallable(int i,MostRelevant mr) {
        this.i = i;
        this.mr = mr;
    }

    @Override
    public TObjectDoubleHashMap<String> call() throws Exception {
        //System.out.println(i);
        double[] similarities = new double[mr.getTrainVectors().size()];
        return mr.processPerTestInstance(i, similarities);
    }
}
