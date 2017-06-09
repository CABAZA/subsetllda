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

import gr.auth.csd.mlkd.LLDACmdOption;
import gr.auth.csd.mlkd.mlclassification.labeledlda.SubsetLLDA;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

/**
 *
 * @author Yannis Papanikolaou perform 5 fold CV on a given trainingSet
 *
 * split jsons iteratively train, split into 100000 instances chunks and
 * predict, concat bips concatenate bips
 *
 */
public class TenFoldCV {

    public static void main(String args[]) throws IOException {
        LLDACmdOption option = new LLDACmdOption(args);
        List<String> instances = Files.readAllLines(new File(option.trainingFile).toPath());

        BufferedReader br = new BufferedReader(new FileReader("C:\\readFile.txt"));
        String[] fold = br.readLine().split(" ");
        
        //split docs
        for (int i = 0; i < 10; i++) {

            for (int j = 0; j < 5; j++) {
                if (j != i) {
                    docs2.putAll(docs.get(j));
                }
            }
            option.trainingFile = "train";
            option.validationFile = "test";
            option.niters = 55;
            option.chains = 1;
            SubsetLLDA mlc = new SubsetLLDA(option);
            mlc.train();
            mlc.predict();
        }

    }
}
