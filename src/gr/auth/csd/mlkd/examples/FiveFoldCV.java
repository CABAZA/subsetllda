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

import gr.auth.csd.mlkd.atypon.CmdOption;
import gr.auth.csd.mlkd.atypon.LLDACmdOption;
import gr.auth.csd.mlkd.atypon.evaluation.EvaluateAll;
import gr.auth.csd.mlkd.atypon.mlclassification.BinaryClassifier;
import gr.auth.csd.mlkd.atypon.mlclassification.MLClassifier;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.SubsetLLDA;
import gr.auth.csd.mlkd.atypon.mlclassification.svm.BinaryRelevanceSVM;
import gr.auth.csd.mlkd.atypon.mlclassification.svm.MetaLabeler;
import gr.auth.csd.mlkd.atypon.mlclassification.svm.SVM;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Document;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.utils.ConcatenateJSONS;
import gr.auth.csd.mlkd.atypon.utils.Timer;
import gr.auth.csd.mlkd.atypon.utils.Utils;
import java.io.File;
import java.util.ArrayList;
import java.util.TreeMap;

/**
 *
 * @author Yannis Papanikolaou perform 5 fold CV on a given trainingSet
 *
 * split jsons iteratively train, split into 100000 instances chunks and
 * predict, concat bips concatenate bips
 *
 */
public class FiveFoldCV {

    public static void main(String args[]) {
        Timer timer = new Timer();

        CmdOption option = new CmdOption(args);
        Dictionary dic;
        CorpusJSON corpus = new CorpusJSON(option.trainingFile);

        dic = new Dictionary(corpus, option.lowUnigrams, option.highUnigrams,
                option.lowBigrams, option.highBigrams);
        Labels labels = new Labels(corpus);
        labels.writeLabels(option.labels);
        dic.writeDictionary(option.dictionary);

        //split docs
        ArrayList<TreeMap<Integer, Document>> docs = ConcatenateJSONS.splitDocsIntofolds(option.trainingFile, 5);
        for (int i = 0; i < 5; i++) {
            ConcatenateJSONS.writeJson(docs.get(i), "test");
            TreeMap<Integer, Document> docs2 = new TreeMap<>();
            for (int j = 0; j < 5; j++) {
                if (j != i) {
                    docs2.putAll(docs.get(j));
                }
            }
            ConcatenateJSONS.writeJson(docs2, "train");
            option.trainingFile = "train";
            option.validationFile = "test";
            trainNpredictFromAll(option, args, i);
            BinaryClassifier.setPredictions(null);
            SVM.setTest(null);
        }
        Utils.concatFiles("bipsValidation","bipsValidation/vanilla");
        Utils.concatFiles("bipsValidation", "bipsValidation/meta");
        Utils.concatFiles("bipsValidation","bipsValidation/tuned");
        Utils.concatFiles("bipsValidation","bipsValidation/subsetllda");

     EvaluateAll ea = new EvaluateAll(option.labels, option.trainingFile, "meta");
    }

    public static void doEverything(MLClassifier mlc, String bipartitionsFile,
            String labels, String testFile, String modelName, boolean svm) {
        mlc.train();
        mlc.predict(null);
        mlc.bipartitionsWrite(bipartitionsFile);
        EvaluateAll ea = new EvaluateAll(labels, testFile, bipartitionsFile);
        File dir = new File("bipsValidation");
        if (!dir.exists()) {
            dir.mkdir();
        }
        Utils.move("bipartitions", "bipsValidation/" + modelName);
        if (svm) {
            Utils.rename("models", "models" + modelName);
        }
        BinaryClassifier.setPredictions(null);
        SVM.setTest(null);

    }

    public static void trainNpredictFromAll(CmdOption option, String[] args, int i) {
        MLClassifier mlc;

        mlc = new BinaryRelevanceSVM(option.trainingFile, option.validationFile,
                option.dictionary, option.labels, option.modelsDirectory, option.threads,
                option.doc2vec, false);
        doEverything(mlc, option.bipartitionsFile, option.labels, option.validationFile, "vanilla"+i, true);

        mlc = new BinaryRelevanceSVM(option.trainingFile, option.validationFile,
                option.dictionary, option.labels, option.modelsDirectory, option.threads,
                option.doc2vec, true);
        doEverything(mlc, option.bipartitionsFile, option.labels, option.validationFile, "tuned"+i, true);
        mlc = new MetaLabeler(option.metalabelerFile, option.trainingFile, option.validationFile,
                option.dictionary, option.labels, option.modelsDirectory, option.threads, option.doc2vec);
        doEverything(mlc, option.bipartitionsFile, option.labels, option.validationFile, "meta" + i, true);
        LLDACmdOption option2 = new LLDACmdOption(args);
        option2.testFile = option.validationFile;
        mlc = new SubsetLLDA(option2);
        doEverything(mlc, option.bipartitionsFile, option.labels, option.validationFile, "subsetllda"+i, false);

    }
}
