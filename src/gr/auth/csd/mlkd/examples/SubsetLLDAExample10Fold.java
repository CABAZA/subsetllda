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

import gnu.trove.map.hash.TObjectDoubleHashMap;


import gr.auth.csd.mlkd.mlclassification.MLClassifier;
import gr.auth.csd.mlkd.mlclassification.labeledlda.SubsetLLDA;
import gr.auth.csd.mlkd.utils.CmdOption;
import gr.auth.csd.mlkd.utils.LLDACmdOption;
import gr.auth.csd.mlkd.utils.Timer;
import java.util.TreeMap;

/**
 *
 * @author Yannis Papanikolaou
 */
public class SubsetLLDAExample10Fold {

    public static void main(String args[]) {
        Timer timer = new Timer();

        CmdOption option = new CmdOption(args);
        LLDACmdOption option2 = new LLDACmdOption(args);
        String tr = option.trainingFile;
        String test = option.testFile;
//        for (int i = 0; i < 10; i++) {
//            option.trainingFile = tr + "" + i;
//            option2.trainingFile = tr + "" + i;
//            option.testFile = test + "" + i;
//            option2.testFile = test + "" + i;

            perFile(option, timer, option2);
        //}

    }

    private static void perFile(CmdOption option, Timer timer, LLDACmdOption option2) {
        
        MLClassifier mlc;

        mlc = new SubsetLLDA(option2);
        Timer trainingTime = new Timer();
        mlc.train();
        System.out.println(trainingTime.durationSeconds());
        //mlc.predict(null);
        Timer testingTime = new Timer();
        mlc.predict();
        System.out.println(testingTime.durationSeconds());
        //Utils.writeObject(probabilities, "probsfile");
    }
}
