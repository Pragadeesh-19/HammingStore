package io.hammingstore.spring.watch;

@FunctionalInterface
public interface HammingWatchListener {
    void onMatch(WatchMatch match);
}
