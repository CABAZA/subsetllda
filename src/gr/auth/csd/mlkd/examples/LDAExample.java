/*
 * Copyright (C) 2016 Yannis Papanikolaou <ypapanik@csd.auth.gr>
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

import gr.auth.csd.mlkd.atypon.LDACmdOption;
import gr.auth.csd.mlkd.atypon.lda.LDA;
import gr.auth.csd.mlkd.atypon.lda.LDACGS_p;
import gr.auth.csd.mlkd.atypon.lda.models.InferenceModel;
import gr.auth.csd.mlkd.atypon.lda.models.Model;
import java.io.File;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
public class LDAExample {

    public static void main(String args[]) {
        LDACmdOption option = new LDACmdOption(args);
        option.K = 200;
        option.alpha = 0.1;
        option.niters = 200;
        LDACGS_p lda = new LDACGS_p(option);
        lda.estimation();
        double[][] trainedTheta_p = lda.getTrnModel().getTheta_p();
        File file = new File(option.modelName + ".phi");
        File file3 = new File(option.modelName + ".phi_p");
        boolean success = file3.renameTo(file);
        Model model = lda.inference();
        double[][] testTheta_p = ((InferenceModel) model).getTheta_p();
        
    }

}
