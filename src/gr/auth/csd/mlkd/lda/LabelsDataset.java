package gr.auth.csd.mlkd.lda;

import gnu.trove.list.array.TIntArrayList;
import gr.auth.csd.mlkd.lda.models.Model;


import java.io.Serializable;
import java.util.ArrayList;

public class LabelsDataset extends DatasetBoW implements Serializable {

    public LabelsDataset(Dictionary dic, boolean inf, int K, String trainedPhi, 
            String labs) {
        super(dic, inf, inf, K, trainedPhi);
        labels = Labels.readLabels(labs);
        V = labels.getSize();
        if (!inf) {
            this.K = K;
        } else {
            this.K = Model.readPhi(trainedPhi).length;
        }
    }

    //Creates the dataset by reading data, dictionary, labels, etc   

    @Override
    public void create(Corpus corpus) {
        docs = new ArrayList<>();
        corpus.reset();
        while ((doc = corpus.nextDocument()) != null) {
            TIntArrayList wordIds = new TIntArrayList();
            for (String l : doc.getLabels()) {
                wordIds.add(labels.getIndex(l) - 1);
            }
            setDoc(new Document(wordIds, doc.getId(), null, doc.getJournal()), docs.size());
        }

    }

    @Override
    public String getWord(Integer index) {
        return labels.getLabel(index + 1);
    }
}
