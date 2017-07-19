package gr.auth.csd.mlkd.lda;

import static gr.auth.csd.mlkd.lda.LDA.data;
import gr.auth.csd.mlkd.lda.models.Model;
import gr.auth.csd.mlkd.mlclassification.labeledlda.LabelsDataset;
import gr.auth.csd.mlkd.utils.LLDACmdOption;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.logging.Level;
import java.util.logging.Logger;

public class LDASerial extends LDA {
        double[][][] phi2;

    public LDASerial(LLDACmdOption option) {
        super(option);
        data = new LabelsDataset(option.trainingFile, false, 0, null, option.K);
    }

    public double[][][] estimation2() {
        Model trnModel;
        phi2= new double[chains/2][][];
//        phi2= new double[1][][];
        data.create(true);
        for (int i = 0; i < chains/2; i++) {
            trnModel = new Model(data, a, false, b, perp, niters, nburnin, modelName, samplingLag);
            phi2[0] = trnModel.estimate(true);
        }
        return phi2;
    }

    public double[][][] getPhi2() {
        return phi2;
    }
    
        public static double[][][] readPhi2(String fi2) {
        double[][][] phi2 = null;
        if (fi2 == null) {
            return null;
        }
        try (ObjectInputStream inputPhi = new ObjectInputStream(new FileInputStream(fi2))) {
            phi2 = (double[][][]) inputPhi.readObject();
        } catch (Exception ex) {
            Logger.getLogger(LDASerial.class.getName()).log(Level.SEVERE, null, ex);
        }
        return phi2;
    }

    public void writePhi(String FiFile) {
        try (ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(FiFile))) {
            output.writeObject(this.phi2);
        } catch (IOException e) {
            System.out.println(e);
        }
    }
    
    public static void main(String args[]) {
        LLDACmdOption option = new LLDACmdOption(args);
        LDASerial ldaserial = new LDASerial(option);
        double[][][] phi2 = ldaserial.estimation2();
        ldaserial.writePhi(option.modelName + ".phi2");
    }
    
}
