# ScholarGrid app image. The pipeline runs elsewhere (your PC / CI / a worker) and
# the data folder with the published release is mounted at /app/data
# (see Docs/deployment.md).
#
#   docker build -t scholargrid .
#   docker run -p 8501:8501 -v $PWD/data:/app/data -e SCHOLARGRID_ENV=production scholargrid

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt .
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheel.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && /opt/venv/bin/pip install -r requirements.txt sentry-sdk
# Bake the query-embedding model into the image so cold starts never hit the network.
# It must be the model the served release was built with (meta.json "embedding_model");
# for older MiniLM releases pass --build-arg EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV HF_HOME=/opt/hf
RUN /opt/venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBED_MODEL}')"

FROM python:3.12-slim
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    SCHOLARGRID_ENV=production \
    SCHOLARGRID_CONFIG=configs/deploy.yaml
# UID 1000 matches Hugging Face Spaces and the usual first user on a Linux server,
# so a mounted data folder stays writable (papers found by live search are saved there).
RUN useradd --create-home --uid 1000 scholargrid
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf
WORKDIR /app
COPY --chown=scholargrid:scholargrid scholargrid ./scholargrid
COPY --chown=scholargrid:scholargrid app ./app
COPY --chown=scholargrid:scholargrid configs ./configs
COPY --chown=scholargrid:scholargrid .streamlit ./.streamlit
COPY --chown=scholargrid:scholargrid app.py ./
RUN mkdir -p data && chown scholargrid:scholargrid data
USER scholargrid
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
