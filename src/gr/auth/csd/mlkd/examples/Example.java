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

import gr.auth.csd.mlkd.atypon.CmdOption;
import gr.auth.csd.mlkd.atypon.LLDACmdOption;
import gr.auth.csd.mlkd.atypon.evaluation.EvaluateAll;
import gr.auth.csd.mlkd.atypon.mlclassification.MLClassifier;
import gr.auth.csd.mlkd.atypon.mlclassification.SimpleModel;
import gr.auth.csd.mlkd.atypon.mlclassification.deeplearning.AutoEncoderModel;
import gr.auth.csd.mlkd.atypon.mlclassification.deeplearning.BinaryRelevanceDBN;
import gr.auth.csd.mlkd.atypon.mlclassification.deeplearning.BinaryRelevanceMLP;
import gr.auth.csd.mlkd.atypon.mlclassification.deeplearning.MultiLabelDBN;
import gr.auth.csd.mlkd.atypon.mlclassification.deeplearning.MultiLabelNN;
import gr.auth.csd.mlkd.atypon.mlclassification.deeplearning.ParagraphVectorModel;
import gr.auth.csd.mlkd.atypon.mlclassification.homer.ClusteringDataset;
import gr.auth.csd.mlkd.atypon.mlclassification.homer.Homer;
import gr.auth.csd.mlkd.atypon.mlclassification.homer.HomerCmdOption;
import gr.auth.csd.mlkd.atypon.mlclassification.homer.LLDAClusteringDataset;
import gr.auth.csd.mlkd.atypon.mlclassification.homer.Tree;
import gr.auth.csd.mlkd.atypon.mlclassification.homer.clusterer.Dbscan;
import gr.auth.csd.mlkd.atypon.mlclassification.homer.clusterer.Hierarchical;
import gr.auth.csd.mlkd.atypon.mlclassification.homer.clusterer.Optics;
import gr.auth.csd.mlkd.atypon.mlclassification.homer.clusterer.RecursiveBalancedKMeans;
import gr.auth.csd.mlkd.atypon.mlclassification.homer.clusterer.RecursiveLabelClustering;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.LLDA;
import gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.SubsetLLDA;
import gr.auth.csd.mlkd.atypon.mlclassification.svm.BinaryRelevanceSVM;
import gr.auth.csd.mlkd.atypon.mlclassification.svm.MetaLabeler;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.utils.Timer;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Yannis Papanikolaou
 */
public class Example {

    public static void main(String args[]) {
        Timer timer = new Timer();

        CmdOption option = new CmdOption(args);
        Dictionary dic = null;
        CorpusJSON corpus = new CorpusJSON(option.trainingFile);
//        if (option.doc2vectrainingFile != null && !(new File("PV.model")).exists()) {
//            ParagraphVectorModel pvm = null;
//            CorpusJSON corpus2 = new CorpusJSON(option.doc2vectrainingFile);
//            Labels labels2 = new Labels(corpus2);
//            labels2.writeLabels(option.labels);
//            dic = new Dictionary(corpus2, option.lowUnigrams, option.highUnigrams,
//                    option.lowBigrams, option.highBigrams);
//            int dimensions = 1000;
//            if (option.doc2vec == 1 || option.doc2vec == 3) {
//                pvm = new ParagraphVectorModel(corpus2, dimensions);
//            } else if (option.doc2vec == 2) {
//                pvm = new AutoEncoderModel(corpus2, dimensions, dic.getId().size(), dic, labels2);
//            }
//            pvm.train();
//            pvm.save("PV.model");
//        } else {
//        dic = new Dictionary(corpus, option.lowUnigrams, option.highUnigrams,
//                option.lowBigrams, option.highBigrams);
//        }

        Labels labels = new Labels(corpus);
        labels.writeLabels(option.labels);
//        dic.writeDictionary(option.dictionary);
//
//        MLClassifier mlc = null;
//        if ("br".equals(option.mlc)) {
//            mlc = new BinaryRelevanceSVM(option.trainingFile, option.testFile,
//                    option.dictionary, option.labels, option.modelsDirectory, option.threads,
//                    option.doc2vec, false);
//        } else if ("tuned".equals(option.mlc)) {
//            mlc = new BinaryRelevanceSVM(option.trainingFile, option.testFile,
//                    option.dictionary, option.labels, option.modelsDirectory, option.threads,
//                    option.doc2vec, true);
//        } else if ("meta".equals(option.mlc)) {
//            mlc = new MetaLabeler(option.metalabelerFile, option.trainingFile, option.testFile,
//                    option.dictionary, option.labels, option.modelsDirectory, option.threads, option.doc2vec);
//        } else if ("mlp".equals(option.mlc)) {
//            mlc = new BinaryRelevanceMLP(option.trainingFile, option.testFile,
//                    option.dictionary, option.labels, option.modelsDirectory, option.threads);
//        } else if ("dbn".equals(option.mlc)) {
//            mlc = new BinaryRelevanceDBN(option.trainingFile, option.testFile,
//                    option.dictionary, option.labels, option.modelsDirectory, option.threads);
//        } else if ("llda".equals(option.mlc)) {
//            LLDACmdOption option2 = new LLDACmdOption(args);
//            mlc = new SubsetLLDA(option2);
//        } else if ("mlnn".equals(option.mlc)) {
//            mlc = new MultiLabelNN(option.trainingFile, option.testFile,
//                    option.dictionary, option.labels, option.modelsDirectory,
//                    option.threads, option.metalabelerFile);
//        } else if ("mldbn".equals(option.mlc)) {
//            mlc = new MultiLabelDBN(option.trainingFile, option.testFile,
//                    option.dictionary, option.labels, option.modelsDirectory,
//                    option.threads, option.metalabelerFile);
//        } else if ("simple".equals(option.mlc)) {
//            mlc = new SimpleModel(option.dictionary, option.labels, option.testFile);
//
//        } else if ("homer".equals(option.mlc)) {
//            HomerCmdOption option3 = new HomerCmdOption(args);
//            String cm = option3.vectorMethod;
//            String clusterer = option3.clusteringMethod;
//            String hmethod = option3.hierarchicalMethod;
//            Homer.baseClassifier = option3.classifier;
//            double epsilon = option3.epsilon;
//
//            String df = option3.distanceFunction;
//            try {
//                ClusteringDataset cd = createHierarchy(cm, option3, args, labels, clusterer, hmethod, df, epsilon);
//            } catch (FileNotFoundException ex) {
//                Logger.getLogger(Example.class.getName()).log(Level.SEVERE, null, ex);
//            }
//            mlc = new Homer(option3);
//        }
//
//        mlc.train();
////        labels.print();
//        mlc.predict(null);
////        mlc.predictProbs(null);
////        Utils.writeObject(probabilities, "probsfile");
//        mlc.bipartitionsWrite(option.bipartitionsFile);
option.bipartitionsFile = "bipartitionsNN";
        EvaluateAll ea = new EvaluateAll(labels, option.testFile, option.bipartitionsFile/*, "probsfile"*/);
//        ev = new MicroAndMacroFLabelPivoted(labels, new CorpusJSON(option.testFile), option.bipartitionsFile+".theta");
//        ev.evaluate();
//        
//        ev = new MicroAndMacroFLabelPivoted(labels, new CorpusJSON(option.testFile), option.bipartitionsFile+".probs");
//        ev.evaluate();
//        
//        ev = new MicroAndMacroFLabelPivoted(labels, new CorpusJSON(option.testFile), option.bipartitionsFile+".probs2");
//        ev.evaluate();
//        
//        ev = new MicroAndMacroFLabelPivoted(labels, new CorpusJSON(option.testFile), option.bipartitionsFile+".voteZ");
//        ev.evaluate();        
    }
    
    
    private static ClusteringDataset createHierarchy(String cm, HomerCmdOption option, String[] args,
            Labels labels, String clusterer, String hmethod, String df, double epsilon) throws FileNotFoundException {
        ClusteringDataset cd;
        if ("cd".equals(cm)) {
            cd = new ClusteringDataset(labels, option.trainingFile, option.labels);
        } else {
            LLDACmdOption option2 = new LLDACmdOption(args);
            cd = new LLDAClusteringDataset(labels, option.trainingFile, option.labels, option2);
        }

        String libSVMFile = cd.writeToFile(option.trainingFile, null);

        System.out.println(clusterer);
        RecursiveLabelClustering cc;

        switch (clusterer) {
            case "kmeans":
                cc = new RecursiveBalancedKMeans(labels, option.maxClusterSize, option.numOfClusters, df);
                break;
            case "optics":
                cc = new Optics(labels, option.maxClusterSize, epsilon, df);
                break;
            case "hierarchical":
                cc = new Hierarchical(labels, option.maxClusterSize, option.numOfClusters, hmethod, df);
                break;
            default:
                cc = new Dbscan(labels, 3, epsilon, df);
                break;
        }
//NOT WORKING        cc = new Clique(cd.getLabels(), option.maxClusterSize, option.numOfClusters);
        cc.recursion(true, cc.hierarchy.getRoot(), libSVMFile, cd, new CorpusJSON(option.trainingFile));
        cc.hierarchy.writeTree(option.treeFile);
        cc.hierarchy.getRoot().print(cc.labels);
        System.out.println("The hierarchy has " + Tree.getNumberOfNodes() + " nodes.");
        return cd;
    }

}
