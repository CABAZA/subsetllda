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
package gr.auth.csd.mlkd.examples;

import gr.auth.csd.mlkd.mlclassification.labeledlda.SubsetLLDA;
import gr.auth.csd.mlkd.utils.Timer;
import gr.auth.csd.mlkd.utils.LLDACmdOption;

/**
 *
 * @author Yannis Papanikolaou
 */
public class SubsetLLDAfExample {

    public static void main(String args[]) {
        Timer timer = new Timer();
        LLDACmdOption option2 = new LLDACmdOption(args);
        
//        option2.trainingFile = "eurlex_train.txt";
//        option2.testFile = "eurlex_test.txt";
        option2.K = 3993;
        //option2.parallel=true;
        SubsetLLDA mlc = new SubsetLLDA(option2);
//        mlc.train();
        mlc.predict();   
//        mlc.predictProbs2(null);
    }
}