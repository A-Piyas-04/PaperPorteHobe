#!/usr/bin/env bash
# Container entry point: optional Streamlit secrets from an env var, DB migrations, then the app.
set -euo pipefail
if [[ -n "${STREAMLIT_SECRETS_TOML_B64:-}" ]]; then
  mkdir -p .streamlit
  echo "$STREAMLIT_SECRETS_TOML_B64" | base64 -d > .streamlit/secrets.toml
  chmod 600 .streamlit/secrets.toml
fi
python -m scholargrid.persistence upgrade
exec streamlit run app/streamlit_app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0
