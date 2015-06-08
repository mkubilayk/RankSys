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

import static es.uam.eps.ir.ranksys.core.util.parsing.DoubleParser.ddp;
import static es.uam.eps.ir.ranksys.core.util.parsing.Parsers.lp;
import static es.uam.eps.ir.ranksys.core.util.parsing.Parsers.sp;
import static es.uam.eps.ir.ranksys.core.util.parsing.Parsers.vp;

import java.util.HashMap;
import java.util.Map;

import es.uam.eps.ir.ranksys.core.feature.FeatureData;
import es.uam.eps.ir.ranksys.core.feature.SimpleFeatureData;
import es.uam.eps.ir.ranksys.core.format.RecommendationFormat;
import es.uam.eps.ir.ranksys.core.format.SimpleRecommendationFormat;
import es.uam.eps.ir.ranksys.core.preference.ConcatPreferenceData;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import es.uam.eps.ir.ranksys.diversity.binom.BinomialModel;
import es.uam.eps.ir.ranksys.diversity.binom.metrics.BinomialDiversity;
import es.uam.eps.ir.ranksys.diversity.distance.metrics.EILD;
import es.uam.eps.ir.ranksys.diversity.intentaware.IntentModel;
import es.uam.eps.ir.ranksys.diversity.intentaware.metrics.ERRIA;
import es.uam.eps.ir.ranksys.diversity.other.metrics.SRecall;
import es.uam.eps.ir.ranksys.metrics.RecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.SystemMetric;
import es.uam.eps.ir.ranksys.metrics.basic.AverageRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.basic.NDCG;
import es.uam.eps.ir.ranksys.metrics.basic.Precision;
import es.uam.eps.ir.ranksys.metrics.rank.NoDiscountModel;
import es.uam.eps.ir.ranksys.metrics.rank.RankingDiscountModel;
import es.uam.eps.ir.ranksys.metrics.rel.BinaryRelevanceModel;
import es.uam.eps.ir.ranksys.metrics.rel.NoRelevanceModel;
import es.uam.eps.ir.ranksys.novdiv.distance.CosineFeatureItemDistanceModel;
import es.uam.eps.ir.ranksys.novdiv.distance.ItemDistanceModel;

/**
 * Example main of metrics.
 *
 * @author Saúl Vargas (saul.vargas@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RerankerEvaluation {

	public static String RERANKED_FOLDER = "src/main/resources/reranked/";
	public static String RATINGS_FOLDER = "src/main/resources/ratings/";

	public static String FEATURES_FILE = "src/main/resources/features/movies.dat";
	public static String RATINGS_FILE = "src/main/resources/ratings/ratings.dat";
	public static String USER_FILE = "src/main/resources/users/users.dat";
	public static String MOVIES_FILE = "src/main/resources/movies/movies.dat";

	public static String[] RECOMMENDERS = { "random", "poprec", "itemknn", "imf" };
	public static String[] RERANKERS = { "random", "binom", "mmr", "xquad", "combsum" };

	public static void main(String[] args) throws Exception {
		Double threshold = 4.0;

		// obtain results for every 5 set of test-training data
		for (int i = 0; i < 5; i++) {
			final int index = i;
			// USER - ITEM - RATING files for train and test
			PreferenceData<Long, Long, Void> trainData = SimplePreferenceData
					.load(RATINGS_FOLDER + "train." + i, lp, lp, ddp, vp);
			PreferenceData<Long, Long, Void> testData = SimplePreferenceData
					.load(RATINGS_FOLDER + "test." + i, lp, lp, ddp, vp);
			PreferenceData<Long, Long, Void> totalData = new ConcatPreferenceData<>(
					trainData, testData);

			// EVALUATED AT CUTOFF 20
			int cutoff = 20;
			// ITEM - FEATURE file
			FeatureData<Long, String, Double> featureData = SimpleFeatureData
					.load(MOVIES_FILE, lp, sp, v -> 1.0);
			// COSINE DISTANCE
			ItemDistanceModel<Long> dist = new CosineFeatureItemDistanceModel<>(
					featureData);
			// BINARY RELEVANCE
			BinaryRelevanceModel<Long, Long> binRel = new BinaryRelevanceModel<>(
					false, testData, threshold);
			// NO RELEVANCE
			NoRelevanceModel<Long, Long> norel = new NoRelevanceModel<>();
			// NO RANKING DISCOUNT
			RankingDiscountModel disc = new NoDiscountModel();
			// INTENT MODEL
			IntentModel<Long, Long, String> intentModel = new IntentModel<>(
					testData.getUsersWithPreferences(), totalData, featureData);
			// BinomialModel
			BinomialModel<Long, Long, String> binomialModel = new BinomialModel<>(
					true, testData.getUsersWithPreferences(), totalData,
					featureData, 0.5);

			Map<String, SystemMetric<Long, Long>> sysMetrics = new HashMap<>();

			// //////////////////////
			// INDIVIDUAL METRICS //
			// //////////////////////
			Map<String, RecommendationMetric<Long, Long>> recMetrics = new HashMap<>();

			// PRECISION
			recMetrics.put("prec", new Precision<>(cutoff, binRel));
			// nDCG
			recMetrics.put("ndcg", new NDCG<>(cutoff,
					new NDCG.NDCGRelevanceModel<>(false, testData, threshold)));

			// //////////////////////
			// DIVERSITY METRICS //
			// //////////////////////
			recMetrics.put("binom-div", new BinomialDiversity<>(binomialModel,
					featureData, cutoff, norel));
			// // EILD
			recMetrics.put("eild", new EILD<>(cutoff, dist, norel, disc));
			// // ERR-IA
			recMetrics.put("err-ia", new ERRIA<>(cutoff, intentModel,
					new ERRIA.ERRRelevanceModel<>(false, testData, threshold)));
			// S-recall
			recMetrics.put("s-recall",
					new SRecall<>(featureData, cutoff, norel));

			// AVERAGE VALUES OF RECOMMENDATION METRICS FOR ITEMS IN TEST
			recMetrics.forEach((name, metric) -> sysMetrics.put(name,
					new AverageRecommendationMetric<>(metric, false)));

			RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<>(
					lp, lp);

			// now we need to implement for each kind of recommender

			for (String reranker : RERANKERS) {
				for (String recommender : RECOMMENDERS) {
					format.getReader(
							RERANKED_FOLDER + "/" + reranker + "/"
									+ recommender + "/" + i + ".recommendation")
							.readAll()
							.forEach(
									rec -> sysMetrics.values().forEach(
											metric -> metric.add(rec)));

					sysMetrics.forEach((metricName, metric) -> System.out
							.println(index + "," + reranker + ","
									+ recommender + "," + metricName + ","
									+ metric.evaluate()));
					sysMetrics.forEach((metricName, metric) -> metric.reset());
				}
			}
			recMetrics.clear();
			sysMetrics.clear();
		}

	}
}
