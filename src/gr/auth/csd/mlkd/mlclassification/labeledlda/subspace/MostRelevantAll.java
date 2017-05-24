package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace;

import gnu.trove.iterator.TDoubleIterator;
import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.iterator.TObjectDoubleIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.set.hash.THashSet;
import gr.auth.csd.mlkd.atypon.CmdOption;
import gr.auth.csd.mlkd.atypon.lda.LDACGS_p;
import gr.auth.csd.mlkd.atypon.lda.models.InferenceModel;
import gr.auth.csd.mlkd.atypon.lda.models.Model;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Document;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import gr.auth.csd.mlkd.atypon.parsers.Parser;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Labels;
import gr.auth.csd.mlkd.atypon.preprocessing.VectorizeJSON;
import gr.auth.csd.mlkd.atypon.utils.Utils;
import java.util.Iterator;
import org.codehaus.jackson.JsonEncoding;
import org.codehaus.jackson.JsonFactory;
import org.codehaus.jackson.JsonGenerator;

/**
 *
 * @author Yannis Papanikolaou
 *
 * Retrieves the n-most relevaqnt documents and creates a subspace of the labels
 * set to be used for LLDA inference
 */
public class MostRelevantAll extends MostRelevantTfIdf {

    private final CmdOption option;

    public MostRelevantAll(int n, CmdOption option) {
        super(n, option);
        this.option = option;
    }

    @Override
    public ArrayList<TObjectDoubleHashMap<String>> mostRelevant() {

        //System.out.println("Most relevant labels...");
        MostRelevantTfIdf mr = new MostRelevantTfIdf(n,
                option.testFile, option.trainingFile, option.dictionary, option.labels);
        ArrayList<TObjectDoubleHashMap<String>> mostRelevant = mr.mostRelevant();

        if(new File("theta.model").exists()) mr = new MostRelevantLDA(20, option, "theta.model");
        else mr = new MostRelevantLDA(n, option);
        ArrayList<TObjectDoubleHashMap<String>> mostRelevantLDA = mr.mostRelevant();

        mr = new MostRelevantDoc2Vec(n, option);
        ArrayList<TObjectDoubleHashMap<String>> mostRelevantD2V = mr.mostRelevant();
        
        for(int i=0;i<mostRelevant.size();i++) {
            TObjectDoubleHashMap<String> lda = mostRelevantLDA.get(i);
            TObjectDoubleIterator<String> it = lda.iterator();
            while(it.hasNext()) {
                it.advance();
                String label = it.key();
                mostRelevant.get(i).adjustOrPutValue(label, it.value(), it.value());
            }
            TObjectDoubleHashMap<String> d2v = mostRelevantD2V.get(i);
            it = d2v.iterator();
            while(it.hasNext()) {
                it.advance();
                String label = it.key();
                mostRelevant.get(i).adjustOrPutValue(label, it.value(), it.value());
            }
        }
        
        this.writeFile(mostRelevant);
        return mostRelevant;

    }
}
