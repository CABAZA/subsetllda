package gr.auth.csd.mlkd.evaluation;

import java.util.TreeSet;

/**
 *
 * @author Yannis Papanikolaou
 */
public class MicroAndMacroF extends Evaluator {

    int numLabels;
    private final int rcut;

    public MicroAndMacroF(String folderTest, String filenamePredictions, int numTags, int rcut) {
        numLabels = numTags;
        this.filenamePredictions = filenamePredictions;
        truth = readTruth(folderTest);
        this.rcut = rcut;
        this.evaluate();

    }

    @Override
    public void evaluate() {
        super.readBipartitions(rcut);
        //load predicted labels
        double[] tp, fp, tn, fn;
        tp = new double[numLabels];
        fp = new double[numLabels];
        tn = new double[numLabels];
        fn = new double[numLabels];

        double df = 0;

        for (int doc = 0; doc < bipartitions.size(); doc++) {
            TreeSet<Integer> t = truth.get(doc);
            TreeSet<Integer> pred = bipartitions.get(doc);
            //System.out.println(doc+" "+t+" "+pred);
            ConfMatrix cm = new ConfMatrix(pred, t, tp, fn, fp, tn, numLabels);
        }

        double macroF = 0;
        double tpa = 0;
        double fpa = 0;
        double tna = 0;
        double fna = 0;
        for (int i = 0; i < numLabels; i++) {
            //System.out.print("Label " + labels.getLabel(i + 1) + " " + (i + 1) + " ");
            //System.out.printf("tp %.0f ", tp[i]);
            tpa += tp[i];
            //System.out.printf("fp %.0f ", fp[i]);
            fpa += fp[i];
            //System.out.printf("tn %.0f ", tn[i]);
            tna += tn[i];
            //System.out.printf("fn %.0f ", fn[i]);
            fna += fn[i];
            double f = 2.0 * tp[i] / (2.0 * tp[i] + fp[i] + fn[i]);
            if (new Double(f).isNaN()) {
                f = 1;
            }
            macroF += f;
            //System.out.printf("f %.2f", f);
            //System.out.println("");
        }

        /*System.out.println(
         "F: " + df / corpus.getCorpusSize());
         */ System.out.println(
                "MacroF: " + macroF / numLabels);
        double microF = 2.0 * tpa / (2.0 * tpa + fpa + fna);

        System.out.println(tpa + ", " + fpa + ", " + fna);
        System.out.println(
                "MicroF: " + microF);

    }

    public static void main(String[] args) {
        String folderTest = args[0];
        String filenamePredictions = args[1];
        int numTags = Integer.parseInt(args[2]);
        int rcut = Integer.parseInt(args[3]);
        MicroAndMacroF ev = new MicroAndMacroF(folderTest, filenamePredictions, numTags, rcut);
    }

}
