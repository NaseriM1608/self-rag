FROM python:3.10-slim

WORKDIR /app

# libgomp1 is required by torch/scipy wheels on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY *.py ./
COPY documents/ documents/
COPY evals/ evals/

EXPOSE 8000

# Point NEO4J_URI at the host running Neo4j Desktop when deploying locally:
#   docker run -e NEO4J_URI=host.docker.internal:7687 ...
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
