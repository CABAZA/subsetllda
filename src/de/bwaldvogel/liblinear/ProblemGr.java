/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.bwaldvogel.liblinear;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author anithagenilos
 */
public class ProblemGr extends Problem {
    public static ProblemGr readFromFile(File file, double bias, int nr_features) throws IOException, InvalidInputDataException {
        return TrainGr.readProblem(file, bias, nr_features);
    }
    
    public static ProblemGr readProblem(String libsvmTestFile, int numFeatures) {
        ProblemGr testProblem = null;
        try {
            testProblem = ProblemGr.readFromFile(new File(libsvmTestFile), 1, numFeatures);
            System.out.println(new Date() + " Finished loading shell");
        } catch (IOException | InvalidInputDataException ex) {
            Logger.getLogger(ProblemGr.class.getName()).log(Level.SEVERE, null, ex);
        }
        return testProblem;
    }

    public static ProblemGr readFromFile(double[][] features, int BIAS, ArrayList<Double> vy) {
        return TrainGr.readProblem(features, BIAS, vy);
    }
}
