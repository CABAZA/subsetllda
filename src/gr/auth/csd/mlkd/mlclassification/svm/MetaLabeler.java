package gr.auth.csd.mlkd.atypon.mlclassification.svm;

import de.bwaldvogel.liblinear.Model;
import static gr.auth.csd.mlkd.atypon.mlclassification.svm.BinaryRelevanceSVM.vectorize;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizeJSON;

import java.io.File;

public class MetaLabeler extends BinaryRelevanceSVM {

    Model metaLabeler;
    private final String metaTrainLabels = "metaTrainLabels";
    private String metalabelerFile;
    private final String trainLibsvm = "train.Libsvm";

    public MetaLabeler(String metaLabelerFile, String trainingFile, String testFile, 
            String dic, String labels, String modelsDirectory, int t, int doc2vec) {
        
        this(metaLabelerFile, (dic == null)? null:Dictionary.readDictionary(dic), 
                (labels == null)?null:Labels.readLabels(labels),
                ((trainingFile != null) ? new CorpusJSON(trainingFile) : null),
                ((testFile != null) ? new CorpusJSON(testFile) : null), modelsDirectory, t,  doc2vec);
    }

    public MetaLabeler(String metaLabelerFile, String trainingFile, String testFile, 
            Dictionary dictionary, Labels labels, String modelsDirectory, int t, int doc2vec) {
        
        this(metaLabelerFile, dictionary, labels, ((trainingFile != null) ? new CorpusJSON(trainingFile) : null),
                ((testFile != null) ? new CorpusJSON(testFile) : null), modelsDirectory, t,  doc2vec);
    }

    public MetaLabeler(String metaLabelerFile, Dictionary dictionary, Labels labels, 
            CorpusJSON trainingCorpus, CorpusJSON testCorpus, String modelsDirectory, int t, int doc2vec) {
        
        super(dictionary, labels, trainingCorpus, testCorpus, modelsDirectory, t, doc2vec, false);
        this.initModel(metaLabelerFile);
        score = true;
        this.doc2vec = doc2vec;
    }

    private void initModel(String metaLabelerFile) {
        if (metaLabelerFile == null) {
            return;
        }
        File f = new File(metaLabelerFile);
        if (f.exists()) {
            metaLabeler = MetaModel.readModel(metaLabelerFile);
        }
        this.metalabelerFile = metaLabelerFile;
    }

    @Override
    public void train() {
        super.train();
        // train meta-labeler
        if (this.metalabelerFile != null) {
            //if(doc2vec==1||doc2vec==2||doc2vec==3) numFeatures+=200;
            vectorize = new VectorizeJSON(dictionary, true, globalLabels);
            vectorize.vectorizeTrain(corpus, "train.Libsvm", "trainLabels", "metaTrainLabels");
            MetaModel ml = new MetaModel(numLabels, trainLibsvm, null, numFeatures);
            ml.train(metaTrainLabels);
            ml.saveModel(metalabelerFile);
            metaLabeler = ml.getModel();
        }
    }

    @Override
    public void createBipartitions() {
        createBipartitionsFromRanking(metalabelerFile);
    }
}
