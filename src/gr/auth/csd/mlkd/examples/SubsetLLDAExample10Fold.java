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

import gnu.trove.map.hash.TObjectDoubleHashMap;
import gr.auth.csd.mlkd.atypon.CmdOption;
import gr.auth.csd.mlkd.atypon.LLDACmdOption;
import gr.auth.csd.mlkd.atypon.evaluation.PrecisionAtK;
import gr.auth.csd.mlkd.atypon.mlclassification.MLClassifier;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.SubsetLLDA;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.utils.Timer;
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
        CorpusJSON corpus = new CorpusJSON(option.trainingFile);
        Labels labels = new Labels(corpus);
        //System.out.println(labels.getLabels());
        labels.writeLabels(option.labels);
        Dictionary dictionary = new Dictionary(corpus, option.lowUnigrams, option.highUnigrams,
                option.lowBigrams, option.highBigrams);
        System.out.println(timer.duration());
        dictionary.writeDictionary(option.dictionary);
        MLClassifier mlc;

        mlc = new SubsetLLDA(option2);
        Timer trainingTime = new Timer();
        mlc.train();
        System.out.println(trainingTime.durationSeconds());
        //mlc.predict(null);
        Timer testingTime = new Timer();
        TreeMap<String, TObjectDoubleHashMap<String>> probabilities = mlc.predictProbs(null);
        System.out.println(testingTime.durationSeconds());
        //Utils.writeObject(probabilities, "probsfile");
        //mlc.createBipartitions();
        //mlc.createBipartitionsFromProbs(option.metalabelerFile, probabilities);
        mlc.bipartitionsWrite(option.bipartitionsFile);

        for (int i = 1; i <= 5; i++) {
            PrecisionAtK ev3 = new PrecisionAtK(i, labels, new CorpusJSON(option.testFile), probabilities);
            ev3.evaluate();
        }

    }
}
