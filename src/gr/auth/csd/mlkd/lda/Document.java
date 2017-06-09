package gr.auth.csd.mlkd.lda;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.set.hash.TIntHashSet;
import java.io.Serializable;

public class Document implements Serializable {

    private final TIntArrayList words;
    TIntDoubleHashMap tfIdfFeatures;
    private final String pmid;
    private final String journal;
    private int[] labels = null;

    Document(TIntDoubleHashMap doc, String id, int[] tags) {
        this.tfIdfFeatures = doc;
        pmid = id;
        labels = tags;
        journal = null;
        words = new TIntArrayList();
        words.addAll(tfIdfFeatures.keySet());
    }

    public TIntDoubleHashMap getTfIdfFeatures() {
        return tfIdfFeatures;
    }

    public TIntArrayList getWords() {
        return words;
    }

    public String getPmid() {
        return pmid;
    }

    public String getJournal() {
        return journal;
    }

    
    public Document(TIntArrayList doc, String id, int[] tlabels, String j)
    {
//        TIntHashSet s = new TIntHashSet();
//        s.addAll(doc);
//        words = new TIntArrayList();
//        words.addAll(s);//
        words = doc;
        pmid = id;
        journal = j;
        if(tlabels != null) labels = tlabels;
    }

    public int[] getLabels() {
        return labels;
    }

}
