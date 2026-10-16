FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Enable bytecode compilation and set Python unbuffered output
ENV UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies (without --frozen so new dependencies in pyproject.toml install cleanly)
COPY pyproject.toml ./
RUN uv sync --no-install-project --no-dev

# Copy application files
COPY . .

# Complete sync including the project
RUN uv sync --no-dev

# Place the virtual environment executables at front of PATH
ENV PATH="/app/.venv/bin:$PATH"

# Default volume directory for mounting local PDF files
RUN mkdir -p /app/data

# Expose Streamlit web UI default port
EXPOSE 8501

# Default command launches the web UI
ENTRYPOINT ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
