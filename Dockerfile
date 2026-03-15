FROM eclipse-temurin:22-jdk-jammy AS builder

WORKDIR /build

COPY pom.xml .
COPY hammingstore-core/pom.xml hammingstore-core/
COPY hammingstore-client/pom.xml hammingstore-client/

RUN apt-get update && apt-get install -y maven && \
    mvn dependency:go-offline -q || true

COPY hammingstore-core/src hammingstore-core/src
COPY hammingstore-client/src hammingstore-client/src

RUN mvn clean package -DskipTests -q && \
    mv hammingstore-core/target/hammingstore-core-*.jar hammingstore-core/target/hammingstore.jar

FROM eclipse-temurin:22-jdk-jammy

WORKDIR /app

COPY --from=builder /build/hammingstore-core/target/hammingstore.jar .

RUN mkdir -p /data

EXPOSE 50051 8080

ENV MAX_VECTORS=500000
ENV DIMS=384
ENV PORT=50051
ENV DATA_DIR=""

ENTRYPOINT ["sh", "-c", \
  "exec java \
    -XX:MaxDirectMemorySize=4g \
    --add-modules jdk.random \
    -jar /app/hammingstore.jar \
    --port=${PORT} \
    --max-vectors=${MAX_VECTORS} \
    --dims=${DIMS} \
    $([ -n \"${DATA_DIR}\" ] && echo \"--data-dir=${DATA_DIR}\" || true)" \
]