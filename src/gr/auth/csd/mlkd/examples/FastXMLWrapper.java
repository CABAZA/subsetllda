/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gr.auth.csd.mlkd.examples;

import java.io.File;
import java.io.IOException;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
public class FastXMLWrapper {

    public static void main(String args[]) throws InterruptedException, IOException {
        eval();
        //runFastXML();
    }

    public static void runFastXML() throws IOException, InterruptedException {
        Process process = new ProcessBuilder("./fastXML.sh", "EUR-Lex").redirectError(new File("err.txt")).redirectOutput(new File("out.txt")).start();
        process.waitFor();
    }
    
    public static void eval() throws IOException, InterruptedException {
        Process process = new ProcessBuilder("./eval.sh", "bibtex", "predictions").redirectError(new File("err.txt")).redirectOutput(new File("out.txt")).start();
        process.waitFor();
    }
    
}
