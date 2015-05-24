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
package es.uam.eps.ir.ranksys.core.preference;

import es.uam.eps.ir.ranksys.core.util.parsing.DoubleParser;
import es.uam.eps.ir.ranksys.core.util.parsing.Parser;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Simple map-based preference data
 *
 * @author Saúl Vargas (saul.vargas@uam.es)
 * 
 * @param <U> type of the users
 * @param <I> type of the items
 * @param <O> type of other information for users and items
 */
public class SimplePreferenceData<U, I, O> implements PreferenceData<U, I, O> {

    private final Map<U, List<IdPref<I, O>>> userMap;
    private final Map<I, List<IdPref<U, O>>> itemMap;
    private final int numPreferences;

    /**
     * Constructor.
     *
     * @param userMap user to preferences map
     * @param itemMap item to preferences map
     * @param numPreferences total number of preferences
     */
    protected SimplePreferenceData(Map<U, List<IdPref<I, O>>> userMap, Map<I, List<IdPref<U, O>>> itemMap, int numPreferences) {
        this.userMap = userMap;
        this.itemMap = itemMap;
        this.numPreferences = numPreferences;
    }

    @Override
    public boolean containsUser(U u) {
        return userMap.containsKey(u);
    }

    @Override
    public int numUsers() {
        return userMap.size();
    }

    @Override
    public int numUsers(I i) {
        return itemMap.getOrDefault(i, new ArrayList<>(0)).size();
    }

    @Override
    public boolean containsItem(I i) {
        return itemMap.containsKey(i);
    }

    @Override
    public int numItems() {
        return itemMap.size();
    }

    @Override
    public int numItems(U u) {
        return userMap.getOrDefault(u, new ArrayList<>()).size();
    }

    @Override
    public int numPreferences() {
        return numPreferences;
    }

    @Override
    public Stream<U> getAllUsers() {
        return userMap.keySet().stream();
    }

    @Override
    public Stream<I> getAllItems() {
        return itemMap.keySet().stream();
    }

    @Override
    public Stream<IdPref<I, O>> getUserPreferences(U u) {
        return userMap.getOrDefault(u, new ArrayList<>()).stream();
    }

    @Override
    public Stream<IdPref<U, O>> getItemPreferences(I i) {
        return itemMap.getOrDefault(i, new ArrayList<>()).stream();
    }

    @Override
    public int numUsersWithPreferences() {
        return userMap.size();
    }

    @Override
    public int numItemsWithPreferences() {
        return itemMap.size();
    }

    @Override
    public Stream<U> getUsersWithPreferences() {
        return userMap.keySet().stream();
    }

    @Override
    public Stream<I> getItemsWithPreferences() {
        return itemMap.keySet().stream();
    }

    /**
     * Load preferences from a file.
     * 
     * Each line is a different preference, with tab-separated fields indicating
     * user, item, weight and other information.
     *
     * @param <U> type of the users
     * @param <I> type of the items
     * @param <O> type of other information
     * @param path path of the input file
     * @param uParser user type parser
     * @param iParser item type parser
     * @param dp double parse
     * @param vParser other info parser
     * @return a simple map-based PreferenceData with the information read
     * @throws IOException when path does not exists of IO error
     */
    public static <U, I, O> SimplePreferenceData<U, I, O> load(String path, Parser<U> uParser, Parser<I> iParser, DoubleParser dp, Parser<O> vParser) throws IOException {
        return load(new FileInputStream(path), uParser, iParser, dp, vParser);
    }

    /**
     * Load preferences from an input stream.
     * 
     * Each line is a different preference, with tab-separated fields indicating
     * user, item, weight and other information.
     *
     * @param <U> type of the users
     * @param <I> type of the items
     * @param <O> type of other information
     * @param in input stream to read from
     * @param uParser user type parser
     * @param iParser item type parser
     * @param dp double parse
     * @param vParser other info parser
     * @return a simple map-based PreferenceData with the information read
     * @throws IOException when path does not exists of IO error
     */
    public static <U, I, O> SimplePreferenceData<U, I, O> load(InputStream in, Parser<U> uParser, Parser<I> iParser, DoubleParser dp, Parser<O> vParser) throws IOException {
        Map<U, List<IdPref<I, O>>> userMap = new HashMap<>();
        Map<I, List<IdPref<U, O>>> itemMap = new HashMap<>();
        int[] numPreferences = new int[]{0};

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(in))) {
            reader.lines().forEach(l -> {
                String[] tokens = l.split("::", 4);
                U user = uParser.parse(tokens[0]);
                I item = iParser.parse(tokens[1]);
                double value;
                if (tokens.length >= 3) {
                    value = dp.parse(tokens[2]);
                } else {
                    value = dp.parse(null);
                }
                O other;
                if (tokens.length == 4) {
                    other = vParser.parse(tokens[3]);
                } else {
                    other = vParser.parse(null);
                }

                numPreferences[0]++;

                List<IdPref<I, O>> uList = userMap.get(user);
                if (uList == null) {
                    uList = new ArrayList<>();
                    userMap.put(user, uList);
                }
                uList.add(new IdPref<>(item, value, other));

                List<IdPref<U, O>> iList = itemMap.get(item);
                if (iList == null) {
                    iList = new ArrayList<>();
                    itemMap.put(item, iList);
                }
                iList.add(new IdPref<>(user, value, other));
            });
        }

        return new SimplePreferenceData<>(userMap, itemMap, numPreferences[0]);
    }

}
