package io.hammingstore.spring.data.repository;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class DerivedQueryParser {

    private static final Pattern RELATION_PATTERN =
            Pattern.compile("^findByRelation(.+?)From$");

    private static final Pattern CHAIN_PATTERN =
            Pattern.compile("^findByChain(.+)$");

    private DerivedQueryParser() {}

    public static DerivedQuery parse(final String methodName) {
        final Matcher relationMatcher = RELATION_PATTERN.matcher(methodName);
        if (relationMatcher.matches()) {
            return DerivedQuery.relation(decapitalise(relationMatcher.group(1)));
        }

        final Matcher chainMatcher = CHAIN_PATTERN.matcher(methodName);
        if (chainMatcher.matches()) {
            final String[] segments  = chainMatcher.group(1).split("And");
            final List<String> relations = new ArrayList<>(segments.length);
            for (final String seg : segments) {
                relations.add(decapitalise(seg));
            }
            if (relations.size() >= 2) {
                return DerivedQuery.chain(relations);
            }
        }

        return null;
    }

    private static String decapitalise(final String s) {
        if (s == null || s.isEmpty()) return s;
        return Character.toLowerCase(s.charAt(0)) + s.substring(1);
    }
}
