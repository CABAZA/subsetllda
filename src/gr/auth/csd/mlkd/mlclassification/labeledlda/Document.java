package gr.auth.csd.mlkd.mlclassification.labeledlda;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import java.io.Serializable;

public class Document implements Serializable {

    private final TIntArrayList words;
    TIntDoubleHashMap tfIdfFeatures;

    private int[] labels = null;

    Document(TIntDoubleHashMap doc, int[] tags) {
        this.tfIdfFeatures = doc;
        labels = tags;
        words = new TIntArrayList();
        words.addAll(tfIdfFeatures.keySet());
    }

    public TIntDoubleHashMap getTfIdfFeatures() {
        return tfIdfFeatures;
    }

    public TIntArrayList getWords() {
        return words;
    }
    
    public Document(TIntArrayList doc, String id, int[] tlabels, String j)
    {
//        TIntHashSet s = new TIntHashSet();
//        s.addAll(doc);
//        words = new TIntArrayList();
//        words.addAll(s);//
        words = doc;
        if(tlabels != null) labels = tlabels;
    }

    public int[] getLabels() {
        return labels;
    }

}
