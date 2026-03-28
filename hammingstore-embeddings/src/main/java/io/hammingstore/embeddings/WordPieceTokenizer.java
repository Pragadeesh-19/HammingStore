package io.hammingstore.embeddings;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WordPieceTokenizer {

    public static final int CLS_ID = 101;
    public static final int SEP_ID = 102;
    public static final int PAD_ID = 0;
    public static final int UNK_ID = 100;

    private static final String CONTINUATION_PREFIX = "##";

    private final Map<String, Integer> vocab;
    private final int maxLength;

    public WordPieceTokenizer(final Path vocabPath, final int maxLength)
            throws IOException {

        this.maxLength = maxLength;
        final List<String> lines = Files.readAllLines(vocabPath,
                java.nio.charset.StandardCharsets.UTF_8);
        this.vocab = new HashMap<>(lines.size() * 2);
        for (int i = 0; i < lines.size(); i++) {
            final String token = lines.get(i).strip();
            if (!token.isEmpty()) {
                this.vocab.put(token, i);
            }
        }
        System.out.printf("[Tokenizer] Loaded %,d tokens from %s%n",
                this.vocab.size(), vocabPath.getFileName());
    }

    public WordPieceTokenizer(final Path vocabPath) throws IOException {
        this(vocabPath, 128);
    }

    public TokenizerOutput tokenize(final String text) {
        final String normalised = normalise(text);

        final String[] words = normalised.split("\\s+");

        final int maxTokens = maxLength - 2; // reserve [CLS] and [SEP]
        final List<Integer> tokenIds = new ArrayList<>(maxTokens);

        for (final String word : words) {
            if (tokenIds.size() >= maxTokens) break;
            if (word.isEmpty()) continue;

            final List<Integer> wordTokens = wordPiece(word);
            for (final int id : wordTokens) {
                if (tokenIds.size() >= maxTokens) break;
                tokenIds.add(id);
            }
        }

        final long[] inputIds = new long[maxLength];
        final long[] attentionMask = new long[maxLength];
        final long[] tokenTypeIds = new long[maxLength];

        inputIds[0] = CLS_ID;
        attentionMask[0] = 1L;

        for (int i = 0; i < tokenIds.size(); i++) {
            inputIds[i + 1] = tokenIds.get(i);
            attentionMask[i + 1] = 1L;
        }

        final int sepPos = tokenIds.size() + 1;
        inputIds[sepPos]      = SEP_ID;
        attentionMask[sepPos] = 1L;

        return new TokenizerOutput(
                reshape(inputIds),
                reshape(attentionMask),
                reshape(tokenTypeIds));
    }

    private List<Integer> wordPiece(final String word) {
        final List<Integer> tokens = new ArrayList<>(4);

        final Integer wholeId = vocab.get(word);
        if (wholeId != null) {
            tokens.add(wholeId);
            return tokens;
        }

        int start = 0;
        boolean badWord = false;

        while (start < word.length()) {
            int end = word.length();
            Integer foundId = null;
            String  found   = null;

            while (start < end) {
                final String substr = start == 0
                        ? word.substring(start, end)
                        : CONTINUATION_PREFIX + word.substring(start, end);

                final Integer id = vocab.get(substr);
                if (id != null) {
                    foundId = id;
                    found = substr;
                    break;
                }
                end--;
            }

            if (foundId == null) {
                badWord = true;
                break;
            }

            tokens.add(foundId);
            start += (found.startsWith(CONTINUATION_PREFIX)
                    ? found.length() - 2
                    : found.length());
        }

        if (badWord) {
            tokens.clear();
            tokens.add(UNK_ID);
        }

        return tokens;
    }

    private static String normalise(final String text) {
        if (text == null || text.isEmpty()) return "";

        final StringBuilder sb = new StringBuilder(text.length() + 16);
        for (int i = 0; i < text.length(); i++) {
            final char c = text.charAt(i);

            if (c == 0 || c == 0xFFFD || isControl(c)) continue;

            if (isWhitespace(c)) {
                sb.append(' ');
                continue;
            }

            if (isPunctuation(c)) {
                sb.append(' ');
                sb.append(Character.toLowerCase(c));
                sb.append(' ');
                continue;
            }

            sb.append(Character.toLowerCase(c));
        }
        return sb.toString().trim();
    }

    private static boolean isWhitespace(final char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r'
                || Character.getType(c) == Character.SPACE_SEPARATOR;
    }

    private static boolean isControl(final char c) {
        if (c == '\t' || c == '\n' || c == '\r') return false;
        final int type = Character.getType(c);
        return type == Character.CONTROL || type == Character.FORMAT
                || type == Character.PRIVATE_USE || type == Character.SURROGATE;
    }

    private static boolean isPunctuation(final char c) {
        if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64)
                || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) {
            return true;
        }
        final int type = Character.getType(c);
        return type == Character.DASH_PUNCTUATION
                || type == Character.START_PUNCTUATION
                || type == Character.END_PUNCTUATION
                || type == Character.CONNECTOR_PUNCTUATION
                || type == Character.OTHER_PUNCTUATION
                || type == Character.INITIAL_QUOTE_PUNCTUATION
                || type == Character.FINAL_QUOTE_PUNCTUATION;
    }

    private static long[][] reshape(final long[] flat) {
        return new long[][]{flat};
    }

    public record TokenizerOutput(
            long[][] inputIds,
            long[][] attentionMask,
            long[][] tokenTypeIds) {}
}
