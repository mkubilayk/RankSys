/* 
 * Copyright (C) 2015 Information Retrieval Group at Universidad Autonoma
 * de Madrid, http://ir.ii.uam.es
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
package es.uam.eps.ir.ranksys.examples;

import es.uam.eps.ir.ranksys.core.feature.FeatureData;
import es.uam.eps.ir.ranksys.core.feature.SimpleFeatureData;
import es.uam.eps.ir.ranksys.core.format.RecommendationFormat;
import es.uam.eps.ir.ranksys.core.format.SimpleRecommendationFormat;
import es.uam.eps.ir.ranksys.core.preference.ConcatPreferenceData;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import static es.uam.eps.ir.ranksys.core.util.parsing.Parsers.lp;
import static es.uam.eps.ir.ranksys.core.util.parsing.Parsers.sp;
import static es.uam.eps.ir.ranksys.core.util.parsing.Parsers.vp;
import static es.uam.eps.ir.ranksys.core.util.parsing.DoubleParser.ddp;
import es.uam.eps.ir.ranksys.novdiv.distance.ItemDistanceModel;
import es.uam.eps.ir.ranksys.novdiv.distance.JaccardFeatureItemDistanceModel;
import es.uam.eps.ir.ranksys.diversity.distance.reranking.MMR;
import es.uam.eps.ir.ranksys.diversity.intentaware.IntentModel;
import es.uam.eps.ir.ranksys.diversity.intentaware.reranking.XQuAD;
import es.uam.eps.ir.ranksys.diversity.binom.BinomialModel;
import es.uam.eps.ir.ranksys.diversity.binom.reranking.BinomialDiversityReranker;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import es.uam.eps.ir.ranksys.novdiv.reranking.Reranker;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Example main of re-rankers.
 *
 * @author Saúl Vargas (saul.vargas@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RerankerExample {
	
	public static String RECOMMENDATIONS_FOLDER = "src/main/resources/recommendations/";
	public static String RATINGS_FOLDER = "src/main/resources/ratings/";
	public static String OUTPUT_FOLDER = "src/main/resources/reranked/";
	
	public static String FEATURES_FILE = "src/main/resources/features/movies.dat";
	public static String RATINGS_FILE = "src/main/resources/ratings/ratings.dat";
	public static String USER_FILE = "src/main/resources/users/users.dat";
	public static String MOVIES_FILE = "src/main/resources/movies/movies.dat";
	
	public static String[] RECOMMENDATIONS = {"random", "poprec", "itemknn", "imf"};

    public static void main(String[] args) throws Exception {
        FastUserIndex<Long> userIndex = SimpleFastUserIndex.load(USER_FILE, lp);
		FastItemIndex<Long> itemIndex = SimpleFastItemIndex.load(MOVIES_FILE, lp);

        FeatureData<Long, String, Double> featureData = SimpleFeatureData.load(FEATURES_FILE, lp, sp, v -> 1.0);
//        FastPreferenceData<Long, Long, Void> recommenderData = SimpleFastPreferenceData
//        		.load(RATINGS_FILE, lp, lp, ddp, vp, userIndex, itemIndex);
        PreferenceData<Long, Long, Void> recommenderData = SimplePreferenceData.load(RATINGS_FILE, lp, lp, ddp, vp);
        
        // ////////////////
 		// RE-RANKERS	 //
 		// ////////////////
        Map<String, Supplier<Reranker<Long, Long>>> rankMap = new HashMap<>();
        
        for (int i = 0; i < RECOMMENDATIONS.length; i++) {
        	PreferenceData<Long, Long, Void> testData = SimplePreferenceData.load(RATINGS_FOLDER + "test." + i, lp, lp, ddp, vp);
        	PreferenceData<Long, Long, Void> trainData = SimplePreferenceData.load(RATINGS_FOLDER + "train." + i, lp, lp, ddp, vp);
        	PreferenceData<Long, Long, Void> totalData = new ConcatPreferenceData<>(trainData, testData);
        	// concat preference data accepts preference data type only.
        	// extend it for fast preference data?
        	
        	final int idx1 = i;
        	
        	for (int j = 0; j < 5; j++) {
        		final int idx2 = i;
        		
        		// MMR diversification
//        		rankMap.put(
//        				OUTPUT_FOLDER + "mmr/" + RECOMMENDATIONS[i] + "/" + j + ".recommendation",
//        				() -> {
//        					double lambda = 0.9;
//            				int cutoff = 20;
//            				
//            				ItemDistanceModel<Long> dist = new JaccardFeatureItemDistanceModel<>(featureData);
//            				return new MMR<>(lambda, cutoff, dist);
//        				});
//        		
        		// xQuAD diversification
        		rankMap.put(
        				OUTPUT_FOLDER + "xquad/" + RECOMMENDATIONS[i] + "/" + j + ".recommendation",
        				() -> {
        					double lambda = 0.2;
            				int cutoff = 20;

            				IntentModel<Long, Long, String> intn = new IntentModel<>(testData.getUsersWithPreferences(), totalData, featureData);
            				return new XQuAD<>(intn, lambda, cutoff, false); // normalize: false?
        				});
        		
				// Binom diversification
//        		rankMap.put(
//        				OUTPUT_FOLDER + "binom/" + RECOMMENDATIONS[i] + "/" + j + ".recommendation",
//        				() -> {
//        					double lambda = 0.7;
//            				int cutoff = 20;
//            				double alpha = 0.5;
//            				
//            				// cache? cached user diversity models?
//            				List<Long> userList = new ArrayList<Long>();
//            				
//            				BinomialModel<Long, Long, String> bin = new BinomialModel<>(false, userList.stream(), recommenderData, featureData, alpha);
//            				return new BinomialDiversityReranker<>(featureData, bin, lambda, cutoff);
//        				});
//        		
        		
        		rankMap.forEach((name, reranker) -> {
        			System.out.println("Running " + name);
        			System.out.flush();
        			
        			String recIn = RECOMMENDATIONS_FOLDER + RECOMMENDATIONS[idx1] + "/" + idx2 + ".recommendation";
        			
        			RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<>(lp, lp);
        			
        			try (RecommendationFormat.Writer<Long, Long> writer = format.getWriter(name)) {
        				 format.getReader(recIn).readAll()
        				 		.map(rec -> reranker.get().rerankRecommendation(rec, 20))
        				 		.forEach(rerankedRecommendation -> {
        				 			try {
        				 				writer.write(rerankedRecommendation);
        				 			} catch (IOException ex) {
        				 				throw new UncheckedIOException(ex);
        				 			}
        				 		});
        				 
        			} catch (IOException ex) {
                        throw new UncheckedIOException(ex);
                    }
        		});
        		
        		rankMap.clear();
        	}
        }
    }
}
