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

import gnu.trove.map.hash.TIntDoubleHashMap;
import gr.auth.csd.mlkd.mlclassification.MLClassifier;
import gr.auth.csd.mlkd.mlclassification.labeledlda.LLDA;
import gr.auth.csd.mlkd.utils.LLDACmdOption;
import gr.auth.csd.mlkd.utils.Timer;
import gr.auth.csd.mlkd.utils.Utils;
import java.util.ArrayList;

/**
 *
 * @author Yannis Papanikolaou
 */
public class MultipleChainsExample {

    public static void main(String args[]) {
        Timer timer = new Timer();
        MLClassifier mlc = null;
        ArrayList<TIntDoubleHashMap> predictions = null;
        LLDACmdOption option2 = new LLDACmdOption(args);
        int chains = 2;

        for (int i = 0; i < chains; i++) {
            mlc = new LLDA(option2);
            mlc.train();
            mlc.predict();

            if (i == 0) {
                predictions = ((LLDA) mlc).getPredictions();
            } else {
                for (int d = 0; d < predictions.size(); d++) {
                    for (int k = 0; k < predictions.get(0).size(); k++) {
                        double th = mlc.getPredictions().get(d).get(k);
                        predictions.get(d).adjustOrPutValue(k, th, th);
                    }
                }
            }
        }

        //normalize
        for (int d = 0; d < predictions.size(); d++) {
            predictions.set(d, Utils.normalize(predictions.get(d), 1.0));

        }
        mlc.setPredictions(predictions);
        mlc.savePredictions();
    }
}
