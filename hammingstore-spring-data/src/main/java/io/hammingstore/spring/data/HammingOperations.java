package io.hammingstore.spring.data;

import io.hammingstore.client.BatchResult;
import io.hammingstore.client.SearchResult;

import java.util.List;

public interface HammingOperations<T, ID> {

    T save(T entity);

    BatchResult saveBatch(List<T> entities);

    default T findById(ID id) {
        throw new UnsupportedOperationException(
                "findById is not supported by HammingStore. "
                        + "HammingStore is a vector search engine — use search() or a VSA query "
                        + "annotation to get entity IDs, then load records from your primary database.");
    }

    List<SearchResult> search(float[] embedding, int k);

    void delete(ID id);

    boolean ping();


}
