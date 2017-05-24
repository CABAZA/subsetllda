/*
 * Copyright (C) 2015 Yannis Papanikolaou
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package gr.auth.csd.mlkd.atypon.examples;

import gr.auth.csd.mlkd.CmdOption;

import gr.auth.csd.mlkd.atypon.evaluation.MicroAndMacroF;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.SubsetLLDATfIdf;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizeMultiLabelLibSVM;
import gr.auth.csd.mlkd.atypon.utils.Timer;
import gr.auth.csd.mlkd.utils.LLDACmdOption;

/**
 *
 * @author Yannis Papanikolaou
 */
public class SubsetLLDAvsLLDATfIdfExample {

    public static void main(String args[]) {
        Timer timer = new Timer();

        CmdOption option = new CmdOption(args);
        LLDACmdOption option2 = new LLDACmdOption(args);
//        mlc = new SubsetLLDA(option2);
//        mlc.train();
//        mlc.predict(null);
//        mlc.bipartitionsWrite(option.bipartitionsFile);
//        EvaluateAll ea = new EvaluateAll(option.labels, option.testFile, option.bipartitionsFile);

        option2.trainingFile = "train.libSVM";
        option2.testFile = "testFile.libSVM";
        
//        option2.trainingFile = "eurlex_train.txt";
//        option2.testFile = "eurlex_test.txt";
        
        
        option2.K = 3993;
        option2.dictionary = null;
        option2.labels = null;
        //option2.parallel=true;
        SubsetLLDATfIdf mlc = new SubsetLLDATfIdf(option2);
        mlc.train();
        mlc.predict(null);
        mlc.bipartitionsWrite(option.bipartitionsFile);
        MicroAndMacroF ea = new MicroAndMacroF(option2.testFile, option.bipartitionsFile, option2.K);
//        
//        mlc.predictProbs2(null);

        //mr.evaluate(option2.testFile, option2.testFile+".wlabels");
        
    }
}