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
package es.uam.eps.ir.ranksys.mf.rec;

import cern.colt.matrix.DoubleMatrix1D;
import es.uam.eps.ir.ranksys.fast.IdxDouble;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.utils.topn.IntDoubleTopN;
import es.uam.eps.ir.ranksys.rec.fast.AbstractFastRecommender;
import es.uam.eps.ir.ranksys.mf.Factorization;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntPredicate;
import java.util.stream.Collectors;

/**
 * Matrix factorization recommender. Scores are calculated as the inner product
 * of user and item vectors.
 *
 * @author Saúl Vargas (saul.vargas@uam.es)
 * 
 * @param <U> type of the users
 * @param <I> type of the items
 */
public class MFRecommender<U, I> extends AbstractFastRecommender<U, I> {

    private final Factorization<U, I> factorization;

    /**
     * Constructor.
     *
     * @param uIndex fast user index
     * @param iIndex fast item index
     * @param factorization matrix factorization
     */
    public MFRecommender(FastUserIndex<U> uIndex, FastItemIndex<I> iIndex, Factorization<U, I> factorization) {
        super(uIndex, iIndex);
        this.factorization = factorization;
    }

    @Override
    public FastRecommendation getRecommendation(int uidx, int maxLength, IntPredicate filter) {
        DoubleMatrix1D pu;

        pu = factorization.getUserVector(uidx2user(uidx));
        if (pu == null) {
            return new FastRecommendation(uidx, new ArrayList<>());
        }

        if (maxLength == 0) {
            maxLength = factorization.numItems();
        }
        IntDoubleTopN topN = new IntDoubleTopN(maxLength);

        DoubleMatrix1D r = factorization.getItemMatrix().zMult(pu, null);
        for (int iidx = 0; iidx < r.size(); iidx++) {
            if (filter.test(iidx)) {
                topN.add(iidx, r.getQuick(iidx));
            }
        }

        topN.sort();
        
        List<IdxDouble> items = topN.reverseStream()
                .map(e -> new IdxDouble(e))
                .collect(Collectors.toList());

        return new FastRecommendation(uidx, items);
    }
}
