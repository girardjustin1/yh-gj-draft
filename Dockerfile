# Always-on deployment of the full engine (Render, Fly.io, Railway, or any Docker host).
#
# The database is rebuilt at startup from public nflverse data — about 8 seconds — so no
# state ships in the image and nothing private is baked in. Your league settings come
# from config/league.yaml, which is gitignored; mount or set it on the host.
#
# FF_ACCESS_TOKEN is required: the server refuses a non-loopback bind without one.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FF_DATA_DIR=/data

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir -e ".[web]"

# Writable volume for the DuckDB file and the nflverse cache.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# Refresh on boot if the database is empty, then serve. A cold start pays ~8s once.
CMD ["sh", "-c", "ff db init && (ff data status | grep -q 'never' && ff data refresh || true) && ff serve --host 0.0.0.0 --port ${PORT:-8000}"]
