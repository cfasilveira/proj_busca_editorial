# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data/samples ./data/samples

RUN uv sync --frozen --no-cache --no-dev \
    && uv run python -m spacy download pt_core_news_sm

# Usuário não privilegiado (segurança)
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/reports \
    && chown -R appuser:appuser /app/data /app/reports
USER appuser

ENTRYPOINT ["editorial"]
CMD ["--help"]
