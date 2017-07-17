package gr.auth.csd.mlkd.mlclassification.labeledlda;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.set.hash.TIntHashSet;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

public class LabelsDataset extends DatasetTfIdf implements Serializable {

    public LabelsDataset(String svmFile, boolean inference, int numFeatures, TIntDoubleHashMap[] fi, int K) {
        super(svmFile, inference, numFeatures, fi);
        V = labels.size();
        this.K = K;
    }

    @Override
    public void create(boolean ignoreFirstLine) {
        docs = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(new File(svmFile)))) {
            String line;
            int id = 0;
            if(ignoreFirstLine) line = br.readLine();
            while ((line = br.readLine()) != null) {
                TIntDoubleHashMap doc = new TIntDoubleHashMap();
                String[] splits = line.split(",");
                TIntHashSet tags = new TIntHashSet();
                for (int i = 0; i < splits.length - 1; i++) {
                    tags.add(Integer.parseInt(splits[i]) + 1);
                }
                String[] splits2 = splits[splits.length - 1].split(" ");
                if (!splits2[0].isEmpty()) {
                    tags.add(Integer.parseInt(splits2[0]) + 1);
                }
                TIntDoubleHashMap ls = new TIntDoubleHashMap();
                TIntIterator it = tags.iterator();
                while (it.hasNext()) {
                    ls.put(it.next(), 1);
                }
                setDoc(new Document(ls, null));
                id++;
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DatasetTfIdf.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(DatasetTfIdf.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    @Override
    public String getWord(Integer index) {
        return super.getLabel(index);
    }
}
