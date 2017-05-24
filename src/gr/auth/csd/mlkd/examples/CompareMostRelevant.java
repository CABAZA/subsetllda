/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gr.auth.csd.mlkd.examples;

import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace.MostRelevantTfIdf;
import gr.auth.csd.mlkd.utils.CmdOption;


/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
public class CompareMostRelevant {
        public static void main(String args[]) {

        CmdOption option = new CmdOption(args);

        MostRelevantTfIdf mr = new MostRelevantTfIdf(10, 
                option.testFile, option.trainingFile, option.dictionary, option.labels);
        mr.mostRelevant();
        MostRelevantTfIdf.evaluate(option.testFile, option.testFile+".wlabels");
//        
//        if(new File("theta.model").exists()) mr = new MostRelevantLDA(10, option, "theta.model");
//        else mr = new MostRelevantLDA(10, option);
//        mr.mostRelevant();
//        MostRelevantTfIdf.evaluate(option.testFile, option.testFile+".wlabels");

//        mr = new MostRelevantAll(10, option);
//        mr.mostRelevant();
//        MostRelevantTfIdf.evaluate(option.testFile, option.testFile+".wlabels");
    }
}
