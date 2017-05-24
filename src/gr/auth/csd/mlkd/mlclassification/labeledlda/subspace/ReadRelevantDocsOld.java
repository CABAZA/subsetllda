/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gr.auth.csd.mlkd.atypon.mlclassification.labeledlda.subspace;

import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.iterator.TObjectIntIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import gnu.trove.set.hash.THashSet;
import gr.auth.csd.mlkd.atypon.mlclassification.svm.MetaModel;
import gr.auth.csd.mlkd.atypon.parsers.Parser;
import gr.auth.csd.mlkd.atypon.preprocessing.CorpusJSON;
import gr.auth.csd.mlkd.atypon.preprocessing.Dictionary;
import gr.auth.csd.mlkd.atypon.preprocessing.Document;
import gr.auth.csd.mlkd.atypon.utils.Pair;
import gr.auth.csd.mlkd.atypon.utils.Utils;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.codehaus.jackson.JsonEncoding;
import org.codehaus.jackson.JsonFactory;
import org.codehaus.jackson.JsonGenerator;

/**
 *
 * @author anithagenilos
 */
public class ReadRelevantDocsOld {

    TIntObjectHashMap<String> idLabels = null;
    protected TIntObjectHashMap<TIntDoubleHashMap> relevantLabelsDistributions;
    protected TObjectIntHashMap<String> labels;

    public ReadRelevantDocsOld(String mapFile, String lsFile) {

        try (ObjectInputStream input = new ObjectInputStream(new FileInputStream(mapFile))) {
            this.relevantLabelsDistributions = (TIntObjectHashMap<TIntDoubleHashMap>) input.readObject();
        } catch (Exception e) {
            System.out.println(e);
        }

        try (ObjectInputStream input = new ObjectInputStream(new FileInputStream(lsFile))) {
            this.labels = (TObjectIntHashMap<String>) input.readObject();
        } catch (Exception e) {
            System.out.println(e);
        }

        idLabels = new TIntObjectHashMap<>();
        TObjectIntIterator<String> it = this.labels.iterator();
        while (it.hasNext()) {
            it.advance();
            idLabels.put(it.value(), it.key());
        }

    }

    protected void writeFile(String testFile, int N) {
        CorpusJSON c = new CorpusJSON(testFile);
        ArrayList<TObjectDoubleHashMap<String>> alpha = new ArrayList<>();
        JsonFactory jfactory = new JsonFactory();
        Document doc;
        try (JsonGenerator jGenerator = jfactory.createJsonGenerator(new File(testFile + ".wlabels"), JsonEncoding.UTF8)) {
            jGenerator.writeStartObject();
            jGenerator.writeFieldName("documents");
            jGenerator.writeStartArray();
            c.reset();
            while ((doc = c.nextDocument()) != null) {
                TIntDoubleHashMap relevantLabels = this.relevantLabelsDistributions.get(Integer.parseInt(doc.getId()));
                THashSet<String> ls = new THashSet<>();
                TIntDoubleIterator it = relevantLabels.iterator();
                while (it.hasNext()) {
                    it.advance();
                    ls.add(this.idLabels.get(it.key()));
                }
                Parser.write(jGenerator, doc.getId(), doc.getTitle(), doc.getAbs(),
                        Integer.toString(doc.getYear()), doc.getJournal(), ls, doc.getBody());
            }
            jGenerator.writeEndArray();
            jGenerator.writeEndObject();
        } catch (IOException ex) {
            Logger.getLogger(ReadRelevantDocsOld.class.getName()).log(Level.SEVERE, null, ex);
        }

        try (ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream("alpha"))) {
            output.writeObject(alpha);
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    private void bipartitionsWrite2(String bipartitionsfirstN, String metalabeler, String testFile, Dictionary dictionary ) {

        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(bipartitionsfirstN)))) {
            double[] metalabelerPredictions = new double[relevantLabelsDistributions.size()];
            if(metalabeler!=null) {
                metalabelerPredictions = MetaModel.getMetaModelPrediction(metalabeler, 
                        dictionary.getId().size(), relevantLabelsDistributions.size(), testFile+".svm", dictionary, new CorpusJSON(testFile));
            }
            
            TIntObjectIterator<TIntDoubleHashMap> iterator = relevantLabelsDistributions.iterator();
            int m=0;
            while (iterator.hasNext()) {
                iterator.advance();
                ArrayList<Pair> wordsProbsList = new ArrayList<>();
                int pmid = iterator.key();
                TIntDoubleIterator iterator1 = iterator.value().iterator();
                while (iterator1.hasNext()) {
                    iterator1.advance();
                    Pair pair = new Pair(iterator1.key(), iterator1.value(), false);
                    wordsProbsList.add(pair);
                }

                Collections.sort(wordsProbsList);
                int N = (int) Utils.round(metalabelerPredictions[m]);
                int iter = (N > wordsProbsList.size()) ? wordsProbsList.size() : N;
                if(iter!=0) writer.write(pmid + ": ");
                for (int i = 0; i < iter; i++) {
                    Integer index = (Integer) wordsProbsList.get(i).first;
                    //if(i<iter-1) writer.write(idLabels.get(index) + "," + wordsProbsList.get(i).second+";"); else writer.write(idLabels.get(index) + "," + wordsProbsList.get(i).second+"\n");
                    if(i<iter-1) writer.write(idLabels.get(index)+"; "); else writer.write(idLabels.get(index)+"\n");
                    /*if( ((Double)wordsProbsList.get(i).second)>=0.8) 
                        writer.write(idLabels.get(index)+"; "); 
                    else {
                        writer.write(idLabels.get(index)+"\n");
                        break;
                    }*/
                }
            }
        }catch (Exception ex) {
                Logger.getLogger(ReadRelevantDocsOld.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    

    public static void main(String args[]) {
        String relevantMap = args[0];
        String relevantLabels = args[1];
        String dictionary = args[2];
        String testFile = args[3];
        String metaLabeler = args[4];
        String bipartitionsFile = args[5];
        ReadRelevantDocsOld rrd = new ReadRelevantDocsOld(relevantMap, relevantLabels);
        //rrd.writeFile(args[2], 0);
        rrd.bipartitionsWrite2(bipartitionsFile, metaLabeler, testFile, Dictionary.readDictionary(dictionary));
//        TIntObjectIterator<TIntDoubleHashMap> it = rrd.relevantLabelsDistributions.iterator();
//        while(it.hasNext()) {
//            it.advance();
//            System.out.print(it.key()+":");
//            for(int i:it.value().keys()) {
//                System.out.print(rrd.idLabels.get(i)+", ");
//            }
//            System.out.println();
//        }
    }
}
