/*
 * Copyright (C) 2015 Yannis Papanikolaou <ypapanik@csd.auth.gr>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package gr.auth.csd.mlkd.atypon.parsers;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.THashSet;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import org.codehaus.jackson.JsonGenerator;

/**
 *
 * @author Yannis Papanikolaou <ypapanik@csd.auth.gr>
 */
public class LibSVMParser extends Parser {
    String in;

    public LibSVMParser(String in) {
        this.in = in;
    }
    
    @Override
    public void createDocList(JsonGenerator jGenerator) throws Exception {
        int doc = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(in))) {
            String line;
            while ((line = br.readLine()) != null) {
                //System.out.println(doc);
                String[] splits = line.split(",");
                ArrayList<String> labels = new ArrayList<>();
                for (int i = 0; i < splits.length - 1; i++) {
                    labels.add(splits[i]);
                }
                String[] splits2 = splits[splits.length - 1].split(" ");
                labels.add(splits2[0]);
                TIntArrayList features = new TIntArrayList();
                for (int i = 1; i < splits2.length; i++) {
                    String[] feat = splits2[i].split(":");
                    for (int j = 0; j < Double.parseDouble(feat[1]); j++) {
                        features.add(Integer.parseInt(feat[0]));
                    }
                }
                doc++;
                String abs = "";
                TIntIterator it = features.iterator();
                while (it.hasNext()) {
                    int f = it.next();
                    abs += f + " ";
                }
                THashSet<String> ls = new THashSet<>();
                ls.addAll(labels);

                Parser.write(jGenerator, doc + "", "", abs, "", "", ls, "");
            }
        }
    }

    public static void main(String args[]) {
        String in = args[0];
        String out = args[1];
        LibSVMParser parser = new LibSVMParser(in);
        parser.parse(out);
    }
}
