package io.hammingstore.embeddings;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Duration;

public final class ModelDownloader {

    private static final String MODEL_NAME    = "all-MiniLM-L6-v2";
    private static final String HF_BASE_URL  =
            "https://huggingface.co/sentence-transformers/"
                    + MODEL_NAME + "/resolve/main/onnx/";

    private static final String VOCAB_URL =
            "https://huggingface.co/sentence-transformers/"
                    + MODEL_NAME + "/resolve/main/vocab.txt";

    /** Minimum expected file sizes — reject clearly truncated downloads. */
    private static final long MIN_MODEL_BYTES = 20_000_000L; // 20 MB
    private static final long MIN_VOCAB_BYTES =   100_000L;  // 100 KB

    private static final Path CACHE_DIR = Paths.get(
            System.getProperty("user.home"),".hammingstore", "models", MODEL_NAME);

    private ModelDownloader() {}

    public static Path ensure() throws IOException, InterruptedException {
        Files.createDirectories(CACHE_DIR);
        downloadIfAbsent("model.onnx", HF_BASE_URL + "model.onnx",  MIN_MODEL_BYTES);
        downloadIfAbsent("vocab.txt",  VOCAB_URL,                    MIN_VOCAB_BYTES);
        return CACHE_DIR;
    }

    public static Path modelPath() throws IOException, InterruptedException {
        return ensure().resolve("model.onnx");
    }

    public static Path vocabPath() throws IOException, InterruptedException {
        return ensure().resolve("vocab.txt");
    }

    private static void downloadIfAbsent(
            final String filename,
            final String url,
            final long   minBytes) throws IOException, InterruptedException {

        final Path target = CACHE_DIR.resolve(filename);

        if (Files.exists(target) && Files.size(target) >= minBytes) {
            System.out.printf("[ModelDownloader] Cache hit: %s (%,d bytes)%n",
                    filename, Files.size(target));
            return;
        }

        System.out.printf("[ModelDownloader] Downloading %s from %s%n", filename, url);

        final HttpClient http = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(30))
                .build();

        final HttpResponse<InputStream> response = http.send(
                HttpRequest.newBuilder()
                        .uri(URI.create(url))
                        .timeout(Duration.ofMinutes(10))
                        .GET()
                        .build(),
                HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() != 200) {
            throw new IOException(
                    "Download failed: HTTP " + response.statusCode() + " from " + url);
        }

        final Path tmp = target.resolveSibling(filename + ".tmp");
        try {
            try (final InputStream  in  = response.body();
                 final OutputStream out = Files.newOutputStream(tmp,
                         StandardOpenOption.CREATE,
                         StandardOpenOption.WRITE,
                         StandardOpenOption.TRUNCATE_EXISTING)) {

                final byte[] buf = new byte[1 << 16]; // 64 KB buffer
                int read;
                long total = 0;
                while ((read = in.read(buf)) != -1) {
                    out.write(buf, 0, read);
                    total += read;
                }
                System.out.printf("[ModelDownloader] Downloaded %s: %,d bytes%n",
                        filename, total);
            }

            final long written = Files.size(tmp);
            if (written < minBytes) {
                throw new IOException(
                        "Download truncated: " + filename
                                + " expected >=" + minBytes + " bytes, got " + written);
            }

            Files.move(tmp, target,
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE);

        } catch (final IOException e) {
            Files.deleteIfExists(tmp);
            throw e;
        }
    }
}
