FROM eclipse-temurin:22-jdk-jammy AS builder

WORKDIR /build
COPY pom.xml .
COPY src ./src

RUN apt-get update && apt-get install -y maven && \
    mvn clean package -DskipTests -q && \
    mv target/hammingstore-core-*.jar target/hammingstore.jar

FROM eclipse-temurin:22-jre-jammy

WORKDIR /app

COPY --from=builder /build/target/hammingstore.jar .

RUN mkdir -p /data

EXPOSE 50051

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
