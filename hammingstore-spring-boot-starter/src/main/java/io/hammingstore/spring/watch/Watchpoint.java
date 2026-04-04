package io.hammingstore.spring.watch;

import java.util.Objects;

public final class Watchpoint {

    private final String id;
    private final String label;
    private final float[] centroid;
    private final double infoThreshold;
    private final double warningThreshold;
    private final double criticalThreshold;

    private Watchpoint(final Builder b) {
        this.id = Objects.requireNonNull(b.id,"id");
        this.label = Objects.requireNonNull(b.label,"label");
        this.centroid = Objects.requireNonNull(b.centroid,"centroid").clone();
        if (this.centroid.length == 0)
            throw new IllegalArgumentException("centroid must not be empty");
        this.infoThreshold = b.infoThreshold;
        this.warningThreshold = b.warningThreshold;
        this.criticalThreshold = b.criticalThreshold;
        if (infoThreshold > warningThreshold || warningThreshold > criticalThreshold)
            throw new IllegalArgumentException(
                    "Thresholds must satisfy: info <= warning <= critical");
    }

    public String id() { return id; }
    public String label() { return label; }
    public float[] centroid() { return centroid.clone(); }
    public int dims() { return centroid.length; }
    public double infoThreshold() { return infoThreshold; }
    public double warningThreshold() { return warningThreshold; }
    public double criticalThreshold() { return criticalThreshold; }

    public WatchSeverity severityFor(final double similarity) {
        if (similarity >= criticalThreshold) return WatchSeverity.CRITICAL;
        if (similarity >= warningThreshold) return WatchSeverity.WARNING;
        if (similarity >= infoThreshold) return WatchSeverity.INFO;
        return null;
    }

    public static Builder builder(final String id) {
        return new Builder(id);
    }

    public static final class Builder {
        private final String id;
        private String label;
        private float[] centroid;
        private double infoThreshold     = 0.60;
        private double warningThreshold  = 0.75;
        private double criticalThreshold = 0.90;

        private Builder(final String id) { this.id = id; }

        public Builder label(final String v) { this.label = v; return this; }
        public Builder centroid(final float[] v) { this.centroid = v; return this; }
        public Builder infoThreshold(final double v) { this.infoThreshold = v; return this; }
        public Builder warningThreshold(final double v) { this.warningThreshold = v; return this; }
        public Builder criticalThreshold(final double v) { this.criticalThreshold = v; return this; }

        /** Sets all three thresholds at t, t+0.10, t+0.20. */
        public Builder threshold(final double t) {
            this.infoThreshold = t;
            this.warningThreshold = t + 0.10;
            this.criticalThreshold = t + 0.20;
            return this;
        }

        public Watchpoint build() { return new Watchpoint(this); }
    }
}
