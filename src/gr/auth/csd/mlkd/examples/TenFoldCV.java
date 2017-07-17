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

import gr.auth.csd.mlkd.mlclassification.labeledlda.LLDA;
import gr.auth.csd.mlkd.utils.LLDACmdOption;
import gr.auth.csd.mlkd.mlclassification.labeledlda.SubsetLLDA;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 *
 * @author Yannis Papanikolaou perform 5 fold CV on a given trainingSet
 *
 * split jsons iteratively train, split into 100000 instances chunks and
 * predict, concat bips concatenate bips
 *
 */
public class TenFoldCV {

    public static void main(String args[]) throws IOException, InterruptedException {
        LLDACmdOption option = new LLDACmdOption(args);
        List<String> instances = Files.readAllLines(new File(option.trainingFile).toPath());
        String[] firstLine = instances.get(0).split(" ");
        int K = Integer.parseInt(firstLine[1]);
        int nrFeatures = Integer.parseInt(firstLine[2]);
        List<String> trainFolds = Files.readAllLines(new File(option.trainFoldsFile).toPath());
        List<String> testFolds = Files.readAllLines(new File(option.testFoldsFile).toPath());
        ArrayList<HashSet<Integer>> trfolds = new ArrayList<>();
        ArrayList<HashSet<Integer>> tstfolds = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            trfolds.add(new HashSet<>());
            tstfolds.add(new HashSet<>());
        }
        for (String fold : trainFolds) {
            String[] f = fold.split(" ");
            for (int i = 0; i < f.length; i++) {
                trfolds.get(i).add(Integer.parseInt(f[i]));
            }
        }
        for (String fold : testFolds) {
            String[] f = fold.split(" ");
            for (int i = 0; i < f.length; i++) {
                tstfolds.get(i).add(Integer.parseInt(f[i]));
            }
        }
        //System.out.println(tstfolds.get(0));

        //create data set and train-predict per fold
        for (int i = 0; i < 1; i++) {
            List<String> train = new ArrayList<>();
            for (int j : trfolds.get(i)) {
                train.add(instances.get(j));
            }
            List<String> test = new ArrayList<>();
            for (int j : tstfolds.get(i)) {
                test.add(instances.get(j));
            }
            option.trainingFile = "train";
            option.testFile = "test";

            write(option.trainingFile, train, K, nrFeatures);
            write(option.testFile, test, K, nrFeatures);
            option.niters = 55;
            option.chains = 1;
            SubsetLLDA mlc = new SubsetLLDA(option);
            //LLDA mlc = new LLDA(option);
            mlc.train();
            mlc.predict();
            Files.move(Paths.get(option.predictionsFile), Paths.get(option.predictionsFile + i));
            Files.move(Paths.get(option.testFile), Paths.get(option.testFile + i));
            Files.delete(Paths.get("test.alpha"));
            Files.delete(Paths.get("test.wlabels"));
            Process process = new ProcessBuilder("./evalSmall.sh", "test"+i, "predictions" + i)
                    //.redirectError(new File("err.txt"))
                    .inheritIO()
                    //.redirectOutput(new File("out.txt"))
                    .start();
            process.waitFor();
        }

    }

    private static void write(String file, List<String> train, int K, int nrFeatures) {
        try (PrintWriter writer = new PrintWriter(file, "UTF-8")) {
            writer.println(train.size()+" "+K+" "+nrFeatures);
            train.stream().forEach((line) -> {
                writer.println(line);
            });
        } catch (IOException e) {
        }
    }
}
