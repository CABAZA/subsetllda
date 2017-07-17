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

import gr.auth.csd.mlkd.mlclassification.labeledlda.DepLLDA;
import gr.auth.csd.mlkd.mlclassification.labeledlda.LLDA;
import gr.auth.csd.mlkd.mlclassification.labeledlda.SubsetLLDA;
import gr.auth.csd.mlkd.utils.Timer;
import gr.auth.csd.mlkd.utils.LLDACmdOption;
import java.io.File;
import java.io.IOException;

/**
 *
 * @author Yannis Papanikolaou
 */
public class SubsetLLDAExample {

    public static void main(String args[]) throws IOException, InterruptedException {
        Timer timer = new Timer();
        LLDACmdOption option2 = new LLDACmdOption(args);

//        option2.trainingFile = "eurlex_train.txt";
//        option2.testFile = "eurlex_test.txt";
//        option2.K = 3993;
//        option2.parallel=true;
        option2.niters = 205;
        option2.chains = 1;
        //SubsetLLDA mlc = new SubsetLLDA(option2);
        LLDA mlc = new LLDA(option2);
        //DepLLDA mlc = new DepLLDA(option2);
        //mlc.train();
        mlc.predict();
        Process process = new ProcessBuilder("./eval.sh", "EUR-Lex", "predictions")
        //Process process = new ProcessBuilder("./eval.sh", "bibtex", "predictions")
                //.redirectError(new File("err.txt")).redirectOutput(new File("out.txt"))
                .inheritIO()
                .start();
        process.waitFor();
    }
}
