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
import static es.uam.eps.ir.ranksys.core.util.parsing.Parsers.vp;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import es.uam.eps.ir.ranksys.core.format.RecommendationFormat;
import es.uam.eps.ir.ranksys.core.format.SimpleRecommendationFormat;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import es.uam.eps.ir.ranksys.mf.Factorization;
import es.uam.eps.ir.ranksys.mf.als.HKVFactorizer;
import es.uam.eps.ir.ranksys.mf.rec.MFRecommender;
import es.uam.eps.ir.ranksys.nn.item.ItemNeighborhoodRecommender;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.CachedItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.TopKItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.VectorCosineItemSimilarity;
import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.PopularityRecommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.RandomRecommender;
import es.uam.eps.ir.ranksys.rec.runner.RecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilterRecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilters;

/**
 * Example main of recommendations.
 *
 * @author Saúl Vargas (saul.vargas@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class RecommenderExample {

	public static String RATINGS_FOLDER = "src/main/resources/ratings/";
	public static String OUTPUT_FOLDER = "src/main/resources/recommendations/";

	public static String USER_FILE = "src/main/resources/users/users.dat";
	public static String MOVIES_FILE = "src/main/resources/movies/movies.dat";

	public static void main(String[] args) throws IOException {

		FastUserIndex<Long> userIndex = SimpleFastUserIndex.load(USER_FILE, lp);
		FastItemIndex<Long> itemIndex = SimpleFastItemIndex.load(MOVIES_FILE,
				lp);

		// ////////////////
		// RECOMMENDERS //
		// ////////////////
		Map<String, Supplier<Recommender<Long, Long>>> recMap = new HashMap<>();

		for (int i = 0; i < 5; i++) {
			// load training and test data
			FastPreferenceData<Long, Long, Void> trainData = SimpleFastPreferenceData
					.load(RATINGS_FOLDER + "train." + i, lp, lp, ddp, vp,
							userIndex, itemIndex);
			FastPreferenceData<Long, Long, Void> testData = SimpleFastPreferenceData
					.load(RATINGS_FOLDER + "test." + i, lp, lp, ddp, vp,
							userIndex, itemIndex);

			
			// random recommender
			final int index = i;
			recMap.put(
					OUTPUT_FOLDER + "random/" + i + ".recommendation",
					() -> {
						return new RandomRecommender<>(trainData, trainData);
					});

			// most-popular recommendation
			recMap.put(
					OUTPUT_FOLDER + "poprec/" + i + ".recommendation",
					() -> {
						return new PopularityRecommender<>(trainData);
					});

			// // item-based nearest neighbors
			recMap.put(
					OUTPUT_FOLDER + "itemknn/" + i + ".recommendation",
					() -> {
						double alpha = 0.5;
						int k = 10;
						int q = 1;

						ItemSimilarity<Long> sim = new VectorCosineItemSimilarity<>(
								trainData, alpha);
						ItemNeighborhood<Long> neighborhood = new TopKItemNeighborhood<>(
								sim, k);
						neighborhood = new CachedItemNeighborhood<>(
								neighborhood);

						return new ItemNeighborhoodRecommender<>(
								trainData, neighborhood, q);
					});

			// // implicit matrix factorization of Hu et al. 2008
			recMap.put(
					OUTPUT_FOLDER + "imf/" + i + ".recommendation",
					() -> {
						int k = 50;
						double lambda = 0.1;
						double alpha = 1.0;
						DoubleUnaryOperator confidence = x -> 1 + alpha * x;
						int numIter = 20;

						Factorization<Long, Long> factorization = new HKVFactorizer<Long, Long>(
								lambda, confidence, numIter).factorize(k,
								trainData);

						return new MFRecommender<>(userIndex, itemIndex,
								factorization);
					});

			// creating recommendation runners
			Set<Long> targetUsers = testData
					.getUsersWithPreferences().collect(Collectors.toSet());
			RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<>(
					lp, lp);
			Function<Long, IntPredicate> filter = FastFilters
					.notInTrain(trainData);
			int maxLength = 100;
			RecommenderRunner<Long, Long> runner = new FastFilterRecommenderRunner<>(
					userIndex, itemIndex, targetUsers, format, filter,
					maxLength);

			// //////////////////////////////
			// GENERATING RECOMMENDATIONS //
			// //////////////////////////////

			recMap.forEach((name, recommender) -> {
				try {
					System.out.println("Running " + name);
					runner.run(recommender.get(), name);
				} catch (IOException ex) {
					throw new UncheckedIOException(ex);
				}
			});

			recMap.clear();
		}

	}
}
