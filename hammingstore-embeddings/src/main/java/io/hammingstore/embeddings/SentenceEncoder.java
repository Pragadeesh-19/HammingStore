package io.hammingstore.embeddings;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

public final class SentenceEncoder implements AutoCloseable {

    public static final int DIMS = 384;
    private static final int MAX_LENGTH = 128;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final WordPieceTokenizer  tokenizer;

    public SentenceEncoder() throws IOException, InterruptedException, OrtException {
        System.out.println("[SentenceEncoder] Initialising...");

        final Path cacheDir = ModelDownloader.ensure();
        final Path modelPath = cacheDir.resolve("model.onnx");
        final Path vocabPath = cacheDir.resolve("vocab.txt");

        this.env = OrtEnvironment.getEnvironment();

        final OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(1);
        opts.setInterOpNumThreads(1);

        this.session   = env.createSession(modelPath.toString(), opts);
        this.tokenizer = new WordPieceTokenizer(vocabPath, MAX_LENGTH);

        System.out.printf("[SentenceEncoder] Ready. Model: %s  Dims: %d%n",
                modelPath.getFileName(), DIMS);
    }

    public float[] encode(final String text) throws OrtException {
        if (text == null || text.isBlank()) {
            return new float[DIMS];
        }

        final WordPieceTokenizer.TokenizerOutput tokens = tokenizer.tokenize(text);

        final long[] shape = {1L, MAX_LENGTH};

        try (final OnnxTensor inputIdsTensor = OnnxTensor.createTensor(
                env, java.nio.LongBuffer.wrap(tokens.inputIds()[0]), shape);
             final OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(
                     env, java.nio.LongBuffer.wrap(tokens.attentionMask()[0]), shape);
             final OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(
                     env, java.nio.LongBuffer.wrap(tokens.tokenTypeIds()[0]), shape)) {

            final Map<String, OnnxTensor> inputs = Map.of(
                    "input_ids", inputIdsTensor,
                    "attention_mask", attentionMaskTensor,
                    "token_type_ids", tokenTypeIdsTensor);

            try (final OrtSession.Result result = session.run(inputs)) {
                final float[][][] hiddenState =
                        (float[][][]) result.get(0).getValue();

                final float[] pooled = meanPool(
                        hiddenState[0], tokens.attentionMask()[0]);

                return l2Normalise(pooled);
            }
        }
    }

    private static float[] meanPool(
            final float[][] hiddenState, final long[] attentionMask) {

        final float[] pooled = new float[DIMS];
        long tokenCount = 0;

        for (int t = 0; t < hiddenState.length; t++) {
            if (attentionMask[t] == 0L) continue;
            tokenCount++;
            for (int d = 0; d < DIMS; d++) {
                pooled[d] += hiddenState[t][d];
            }
        }

        if (tokenCount > 0) {
            final float inv = 1.0f / tokenCount;
            for (int d = 0; d < DIMS; d++) {
                pooled[d] *= inv;
            }
        }

        return pooled;
    }

    private static float[] l2Normalise(final float[] vec) {
        double sumSq = 0.0;
        for (final float v : vec) sumSq += (double) v * v;

        final double norm = Math.sqrt(sumSq);
        if (norm > 1e-12) {
            final float invNorm = (float) (1.0 / norm);
            for (int i = 0; i < vec.length; i++) vec[i] *= invNorm;
        }
        return vec;
    }

    @Override
    public void close() {
        try { session.close(); } catch (final OrtException ignored) {}
        env.close();
    }

    public static void main(final String[] args)
            throws IOException, InterruptedException, OrtException {

        System.out.println("[SentenceEncoder] Running smoke test...");

        try (final SentenceEncoder encoder = new SentenceEncoder()) {
            final String[] texts = {
                    "France",
                    "Germany",
                    "Quantum mechanics",
                    "The Eiffel Tower is located in Paris"
            };

            final float[][] embeddings = new float[texts.length][];
            for (int i = 0; i < texts.length; i++) {
                embeddings[i] = encoder.encode(texts[i]);
                System.out.printf("Encoded: %-45s  norm=%.6f  first5=[%.3f, %.3f, %.3f, %.3f, %.3f]%n",
                        "\"" + texts[i] + "\"",
                        norm(embeddings[i]),
                        embeddings[i][0], embeddings[i][1], embeddings[i][2],
                        embeddings[i][3], embeddings[i][4]);
            }

            System.out.println("\nCosine similarities (dot product of unit vectors):");
            for (int i = 0; i < texts.length; i++) {
                for (int j = i + 1; j < texts.length; j++) {
                    System.out.printf("  %-20s <-> %-45s  sim=%.4f%n",
                            "\"" + texts[i] + "\"",
                            "\"" + texts[j] + "\"",
                            dot(embeddings[i], embeddings[j]));
                }
            }

            System.out.println("\n[SentenceEncoder] Smoke test complete.");
            System.out.println("Expected: France/Germany similarity > France/QuantumMechanics similarity");
        }
    }

    private static float norm(final float[] v) {
        double sum = 0.0;
        for (final float f : v) sum += (double) f * f;
        return (float) Math.sqrt(sum);
    }

    private static float dot(final float[] a, final float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }
}
