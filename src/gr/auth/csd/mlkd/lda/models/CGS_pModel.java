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
package gr.auth.csd.mlkd.lda.models;

import gnu.trove.list.array.TIntArrayList;
import gr.auth.csd.mlkd.mlclassification.labeledlda.DatasetTfIdf;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
public class CGS_pModel extends Model {

    double[][] phi_p;
    double[][] theta_p;

    public double[][] getTheta_p() {
        return theta_p;
    }

    public double[][] getPhi_p() {
        return phi_p;
    }

    public CGS_pModel(DatasetTfIdf data, double a, boolean inf, double b, boolean perp, int niters, int nburnin, String modelName, int sl) {
        super(data, a, inf, b, perp, niters, nburnin, modelName, sl);
        phi_p = new double[K][V];
        theta_p = new double[M][K];
    }

    @Override
    public void updateParams(boolean average, boolean finalSample) {
        numSamples++;
        this.computePhi_p();
        this.computeTheta_p(finalSample);
    }

    protected double[][] computeTheta_p(boolean finalSample) {

        for (int d = 0; d < M; d++) {
            double[] tempTheta = new double[K];
            double[] p = new double[K];

            TIntArrayList words = data.getDocs().get(d).getWords();
            int ppxCounter = 0;
            for (int w = 0; w < words.size(); w++) {
                int word = data.getDocs().get(d).getWords().get(w);
                int topic = z[d][w];
                nd[d][topic]--;
                for (int k = 0; k < K; k++) {
                    p[k] = (alpha[k] + nd[d][k]) * phi_p[k][word];
                }
                nd[d][topic]++;
                //average sampling probabilities
                p = Utils.normalize(p, 1);

                for (int k = 0; k < K; k++) {
                    tempTheta[k] += p[k];
                }
                if (perplexity && ppxCounter >= (words.size() / 2)) {
                    break;
                }
                ppxCounter++;
            }

            for (int k = 0; k < K; k++) {
                theta_p[d][k] += tempTheta[k];
            }
        }

        if (finalSample) {
            for (int d = 0; d < M; d++) {
                //if(d==0||d==1) System.out.println(Arrays.toString(theta_p[d]));
                for (int k = 0; k < K; k++) {
                    theta_p[d][k] /= numSamples;
                    theta_p[d][k] += alpha[k];
                }
                // normalize
                theta_p[d] = Utils.normalize(theta_p[d], 1);
            }
        }
        return theta_p;
    }

    private double[][] computePhi_p() {
        phi_p = new double[K][V];
        for (int d = 0; d < M; d++) {
            double[] p = new double[K];
            TIntArrayList words = data.getDocs().get(d).getWords();
            for (int w = 0; w < words.size(); w++) {
                int word = data.getDocs().get(d).getWords().get(w);
                int topic = z[d][w];
                nd[d][topic]--;
                for (int k = 0; k < K; k++) {
                    p[k] = (nw[k][word] + beta) * (nd[d][k] + alpha[k]) / (nwsum[k] + betaSum);
                }
                nd[d][topic]++;
                //average sampling probabilities
                p = Utils.normalize(p, 1);
                for (int k = 0; k < K; k++) {
                    phi_p[k][word] += p[k];
                }
            }
        }
        for (int k = 0; k < K; k++) {
            //add beta hyperparameter
            for (int w = 0; w < V; w++) {
                phi_p[k][w] = phi_p[k][w] + beta;
            }
            //normalize
            phi_p[k] = Utils.normalize(phi_p[k], 1.0);
        }
        return phi_p;
    }

    protected void savePhi_p(String modelName) {
        try (ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(modelName + ".phi_p"))) {
            output.writeObject(this.phi_p);
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    @Override
    public void save(String modelName, int twords) {
        //bipartitionsWrite("bipartitions." + modelName, twords, theta);
        if (!inference) {
            saveTwords("twords." + modelName, twords);
            this.savePhi(modelName);
            this.savePhi_p(modelName);
        }
        saveTheta("theta." + modelName, theta_p);
    }

    public static double[][] readPhi_p(String fi) {
        double[][] phi_p = null;
        try (ObjectInputStream input = new ObjectInputStream(new FileInputStream(fi))) {
            phi_p = (double[][]) input.readObject();
        } catch (Exception e) {
            System.out.println(e.getCause());
        }
        //System.out.println(fi + " loaded: K= " + phi.length);
        return phi_p;
    }

}
