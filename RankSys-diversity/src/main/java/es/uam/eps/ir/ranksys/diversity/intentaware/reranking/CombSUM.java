package es.uam.eps.ir.ranksys.diversity.intentaware.reranking;

import es.uam.eps.ir.ranksys.core.IdDouble;
import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.diversity.intentaware.IntentModel;
import es.uam.eps.ir.ranksys.novdiv.reranking.LambdaReranker;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;

public class CombSUM<U, I, F> extends LambdaReranker<U, I> {
	
	private final IntentModel<U, I, F> intentModel;
	
	/**
     * Constructor.
     *
     * @param intentModel intent-aware model
     * @param lambda trade-off between novelty and relevance
     * @param cutoff number of items to be greedily selected
     * @param norm normalize the linear combination between relevance and 
     * novelty
     */
	public CombSUM(IntentModel<U, I, F> intentModel, double lambda, int cutoff, boolean norm) {
		super(lambda, cutoff, norm);
		this.intentModel = intentModel;
	}
	
	@Override
	protected LambdaUserReranker getUserReranker(Recommendation<U, I> recommendation, int maxLength) {
		return new UserCombSUM(recommendation, maxLength);
	}
	
	protected class UserCombSUM extends LambdaUserReranker {
		
		private final IntentModel<U, I, F>.UserIntentModel uim;
        private final Object2DoubleOpenHashMap<F> redundancy;
        private final Object2DoubleOpenHashMap<F> probNorm;
        
        /**
         * Constructor.
         *
         * @param recommendation input recommendation to be re-ranked
         * @param maxLength maximum length to be re-ranked with CombSUM
         */
        
        public UserCombSUM(Recommendation<U, I> recommendation, int maxLength) {
            super(recommendation, maxLength);

            this.uim = intentModel.getModel(recommendation.getUser());
            this.redundancy = new Object2DoubleOpenHashMap<>();
            this.redundancy.defaultReturnValue(1.0);
            this.probNorm = new Object2DoubleOpenHashMap<>();
            recommendation.getItems().forEach(iv -> {
                uim.getItemIntents(iv.id).sequential().forEach(f -> {
                    probNorm.addTo(f, iv.v);
                });
            });
        }

        private double pif(IdDouble<I> iv, F f) {
            return iv.v / probNorm.getDouble(f);
        }
        
        @Override
        protected double nov(IdDouble<I> iv) {
            return uim.getItemIntents(iv.id)
                    .mapToDouble(f -> {
                        return uim.p(f) * pif(iv, f);
                    }).sum();
        }

        @Override
        protected void update(IdDouble<I> biv) {
            uim.getItemIntents(biv.id).sequential()
                    .forEach(f -> {
                        redundancy.put(f, redundancy.getDouble(f) * (1 - pif(biv, f)));
                    });
        }
	}
}
