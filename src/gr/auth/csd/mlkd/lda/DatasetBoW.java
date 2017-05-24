package gr.auth.csd.mlkd.atypon.lda;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.THashSet;
import gr.auth.csd.mlkd.atypon.lda.models.Model;
import gr.auth.csd.mlkd.atypon.preprocessing.Corpus;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.preprocessing.NGram;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

public class DatasetBoW extends Dataset {

    // number of documents
    protected Dictionary dictionary;
    protected NGram[] ngrams;
    protected HashMap<String, THashSet<String>> labelvalues;
    protected Labels labels = null;

    public DatasetBoW(Dictionary dic, boolean unlabeled, boolean inference,
            int K, String trainedPhi) {
        super(unlabeled, inference);
        dictionary = dic;
        if (!inference) {
            this.K = K;
        } else if (trainedPhi != null) {
            this.K = Model.readPhi(trainedPhi).length;
        }
        V = dictionary.getId().size();
    }

    public DatasetBoW(Dictionary dic, Labels labs, boolean unlabeled, boolean inf) {
        this(dic, unlabeled, inf, 0, null);
        labels = labs;
        this.K = labels.getSize();
    }

    @Override
    public void create(Corpus corpus) {
        gr.auth.csd.mlkd.atypon.preprocessing.Document doc;
        docs = new ArrayList<>();
        corpus.reset();
        while ((doc = corpus.nextDocument()) != null) {
            int[] labelIds = null;
            if (!unlabeled) {
                labelIds = addLabels(doc.getLabels());
            }
            List<String> lines = doc.getContentAsSentencesOfTokens(true);
            TIntArrayList wordIds = new TIntArrayList();
            for (int j = 0; j < dictionary.getNGramSizes().size(); j++) {
                TIntArrayList temp = nGramsFromSentences(lines, dictionary.getNGramSizes().get(j));
                wordIds.addAll(temp);
            }
            setDoc(new Document(wordIds, doc.getId(), labelIds, doc.getJournal()), docs.size());
        }
    }

    //get n-grams from each document
    protected TIntArrayList nGramsFromSentences(List<String> lines, int n) {
        TIntArrayList wordIds = new TIntArrayList();
        for (String line : lines) {
            String[] tokens = line.split(" ");
            for (int i = 0; i < tokens.length + 1 - n; i++) {
                List<String> aList = new ArrayList<>();
                for (int j = 0; j < n; j++) {
                    aList.add(tokens[i + j]);
                }
                NGram ngram = new NGram(aList);
                List<String> list = ngram.getList();

                if (dictionary.getDocumentFrequency().containsKey(ngram)) {
                    wordIds.add(dictionary.getId().get(ngram));
                    //if(inference&&!inferenceDictionary.contains(dictionary.getId().get(ngram))) inferenceDictionary.add(dictionary.getId().get(ngram));
                }
            }
        }
        return wordIds;
    }

    public Dictionary getDictionary() {
        return dictionary;
    }

    @Override
    public String getWord(Integer index) {
        return dictionary.getNgram(index).toString();
    }

    protected int[] addLabels(Set<String> labs) {
        TIntArrayList lids = new TIntArrayList();
        for (String s : labs) {
            int index = labels.getIndex(s);
            if (index != -1) {
                lids.add(index);
            }
        }
        return lids.toArray();
    }

    /**
     *
     * @param docId
     * @param labelsDoc
     * @return
     */
    protected TIntArrayList addLabels(String docId, String labelsDoc) {
        TIntArrayList lids = new TIntArrayList();
        for (String s : labelvalues.get(docId)) {
            int index = labels.getIndex(s);
            if (index != -1) {
                lids.add(index);
            }
        }
        return lids;
    }

    public double[] freq() {
        double[] freq = new double[V];
        double totalWords = 0;
        for (Document d : docs) {
            totalWords += d.getWords().size();
        }
        for (Document d : docs) {
            TIntIterator it = d.getWords().iterator();
            while (it.hasNext()) {
                freq[it.next()]++;
            }
        }
        for (int w = 0; w < V; w++) {
            freq[w] /= totalWords;
        }
        return freq;
    }

    @Override
    public String getLabel(int id) {
        return labels.getLabel(id);
    }

    public Labels getLabels() {
        return labels;
    }
}
