# Tools

_This repo contains tools used for running local LLMs and parsing documents._

---

## Running with Docker

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker engine)
- [Ollama](https://ollama.com/) running with the desired model pulled (e.g. `ollama run medgemma1.5`).

### 1. Place PDFs in the `data/` directory

Drop any PDFs you want to analyze into the local `./data/` folder.

### 2. Build the Docker image

```bash
docker build -t llm-pdf-parser .
```

### 3. Run a query using Docker

```bash
docker run --rm \
  --add-host host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
  -e OLLAMA_MODEL=medgemma1.5 \
  -v ${PWD}/data:/app/data \
  llm-pdf-parser /app/data/sample.pdf "What are the key findings?"
```

---

## Running the Web UI (Streamlit)

You can run the web-based interactive interface using:

```bash
# Using the local virtual environment
.\.venv\Scripts\streamlit run app.py
```

The UI will be accessible in your browser at `http://localhost:8501`.

---

## Running the Web UI with Docker Compose

Start both Ollama and the Streamlit UI parser with a single command:

```bash
docker compose up --build
```

- **Web UI**: Open `http://localhost:8501` in your browser.
- **Ollama API**: Available at `http://localhost:11434`.

---

## Running CLI Query with Docker Compose

Run interactive CLI queries on a file in `data/`:

```bash
docker compose run --rm --entrypoint python pdf-parser pdf-reader.py /app/data/sample.pdf "Your question here"
```

