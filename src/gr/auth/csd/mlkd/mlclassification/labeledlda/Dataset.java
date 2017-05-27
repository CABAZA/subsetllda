/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gr.auth.csd.mlkd.mlclassification.labeledlda;

import java.io.Serializable;
import java.util.ArrayList;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
public abstract class Dataset implements Serializable {
    
    static final long serialVersionUID = 5716568340666075332L;
    protected int V = 0; // number of words
    protected int K;
    public final boolean inference;
    protected ArrayList<Document> docs; // a list of documents				 		// number of documents


    public Dataset(boolean inference) {
        this.inference = inference;
    }



    public abstract void create(boolean ignoreFirstLine);

    public int getV() {
        return V;
    }


    public int getK() {
        return K;
    }

    public ArrayList<Document> getDocs() {
        return docs;
    }

    public void setDoc(Document doc) {
            docs.add(doc);
    }

    public abstract String getLabel(int id);

    public abstract String getWord(Integer index);
    
}
