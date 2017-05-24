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

import gnu.trove.set.hash.THashSet;
import gr.auth.csd.mlkd.atypon.CmdOption;
import gr.auth.csd.mlkd.atypon.LLDACmdOption;
import gr.auth.csd.mlkd.atypon.evaluation.EvaluateAll;
import gr.auth.csd.mlkd.atypon.mlclassification.MLClassifier;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.LLDA;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.LLDAExperimentCGSvsCGS_p;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.utils.Timer;
import java.io.File;
import java.util.TreeMap;

/**
 *
 * @author Yannis Papanikolaou
 */
public class MultipleChainsExample {

    public static void main(String args[]) {
        Timer timer = new Timer();

        CmdOption option = new CmdOption(args);
        CorpusJSON corpus = new CorpusJSON(option.trainingFile);
        Labels labels = new Labels(corpus);
        labels.writeLabels(option.labels);
        Dictionary dictionary = new Dictionary(corpus, option.lowUnigrams, option.highUnigrams,
                option.lowBigrams, option.highBigrams);
        System.out.println(timer.duration());
        dictionary.writeDictionary(option.dictionary);
        MLClassifier mlc = null;

        LLDACmdOption option2 = new LLDACmdOption(args);
        int chains = 2;
        double[][] predictions = null;
        double[][] predictions_p = null;

        double[][] predictionsPhi_p = null;
        double[][] predictions_pPhi_p = null;

        for (int i = 0; i < chains; i++) {
            //mlc = new LLDA(option2);
            mlc = new LLDAExperimentCGSvsCGS_p(option2);
            mlc.train();
            mlc.predictWithoutBipartitions(null);

            if (i == 0) {
                predictions = ((LLDA) mlc).getPredictions();
                predictions_p = ((LLDAExperimentCGSvsCGS_p) mlc).getPredictions_p();
            } else {
                for (int d = 0; d < predictions.length; d++) {
                    for (int k = 0; k < predictions[0].length; k++) {
                        predictions[d][k] += ((LLDA) mlc).getPredictions()[d][k];
                        predictions_p[d][k] += ((LLDAExperimentCGSvsCGS_p) mlc).getPredictions_p()[d][k];
                    }
                }
            }

            File file = new File(option2.trainingFile + ".model.phi");
            File file2 = new File(option2.trainingFile + ".model.phi.std");
            boolean success = file.renameTo(file2);

            File file3 = new File(option2.trainingFile + ".model.phi_p");
            success = file3.renameTo(file);

            mlc = new LLDAExperimentCGSvsCGS_p(option2);
            mlc.predictWithoutBipartitions(null);

            if (i == 0) {
                predictionsPhi_p = ((LLDA) mlc).getPredictions();
                predictions_pPhi_p = ((LLDAExperimentCGSvsCGS_p) mlc).getPredictions_p();
            } else {
                for (int d = 0; d < predictions.length; d++) {
                    for (int k = 0; k < predictions[0].length; k++) {
                        predictionsPhi_p[d][k] += ((LLDA) mlc).getPredictions()[d][k];
                        predictions_pPhi_p[d][k] += ((LLDAExperimentCGSvsCGS_p) mlc).getPredictions_p()[d][k];
                    }
                }
            }

            success = file.renameTo(file3);
            success = file2.renameTo(file);

        }

        //normalize
        for (int d = 0; d < predictions.length; d++) {
            for (int k = 0; k < predictions[0].length; k++) {
                predictions[d][k] /= chains;
                predictions_p[d][k] /= chains;
                predictionsPhi_p[d][k] /= chains;
                predictions_pPhi_p[d][k] /= chains;

            }
        }

        ((LLDA) mlc).setPredictions(predictions);
        mlc.createBipartitionsFromRanking(option.metalabelerFile);
        mlc.bipartitionsWrite(option.bipartitionsFile + ".std");;
        ((LLDA) mlc).setPredictions(predictions_p);
        mlc.setBipartitions(new TreeMap<String, THashSet<String>>());
        mlc.createBipartitionsFromRanking(option.metalabelerFile);
        mlc.bipartitionsWrite(option.bipartitionsFile);
        System.out.println("std phi + theta_p:");
        EvaluateAll ea = new EvaluateAll(labels, option.testFile, option.bipartitionsFile, "probsfile");
        System.out.println("std phi + std theta:");
        ea = new EvaluateAll(labels, option.testFile, option.bipartitionsFile + ".std", "probsfile");
        ((LLDA) mlc).setPredictions(predictionsPhi_p);
        mlc.setBipartitions(new TreeMap<String, THashSet<String>>());
        mlc.createBipartitionsFromRanking(option.metalabelerFile);
        mlc.bipartitionsWrite(option.bipartitionsFile + ".std.phi_p");;
        ((LLDA) mlc).setPredictions(predictions_pPhi_p);
        mlc.setBipartitions(new TreeMap<String, THashSet<String>>());
        mlc.createBipartitionsFromRanking(option.metalabelerFile);
        mlc.bipartitionsWrite(option.bipartitionsFile + ".cgs_p");;
        System.out.println("phi_p + theta_p:");
        ea = new EvaluateAll(labels, option.testFile, option.bipartitionsFile+ ".cgs_p", "probsfile");
        System.out.println("phi_p + std theta:");
        ea = new EvaluateAll(labels, option.testFile, option.bipartitionsFile + ".std.phi_p", "probsfile");

    }
}
