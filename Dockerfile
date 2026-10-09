# ==============================================================================
# chatbot_web — AWS Bedrock AgentCore Runtime image
# Build context: chatbot_web/ directory
#
#   cd chatbot_web
#   docker build -t asl-web-chatbot:v1.0.0 .
# ==============================================================================

# ── Stage 1: install dependencies ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --retries / --timeout make the build tolerant of slow or flaky PyPI reads.
# Without these, a single 15s read timeout on one dependency (e.g.
# charset-normalizer) makes pip backtrack through every version and fail with a
# misleading "ResolutionImpossible" that is really just a network drop.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/deps \
       --retries 10 --timeout 120 \
       -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /deps /usr/local

# Copy application source (config/.env is the single config, baked in)
COPY . /app/

# Non-root user for security
RUN useradd --system --uid 1001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "entrypoint.py"]
