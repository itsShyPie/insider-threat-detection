#!/bin/bash
# entrypoint.sh - container command dispatcher
#
# Supported commands:
#   train    Train all models, evaluate, save artefacts
#   serve    Launch the Streamlit prototype (default)
#   predict  Run a single prediction from stdin (sanity check)
#   shell    Drop into a bash shell inside the container

set -euo pipefail

cmd="${1:-serve}"

case "$cmd" in
    train)
        echo "==> Running training pipeline ..."
        exec python -m model.train
        ;;
    serve)
        # Train first if no model exists yet
        if [ ! -f /app/models/rf_model.pkl ]; then
            echo "==> No trained model found. Running initial training ..."
            python -m model.train
        fi
        echo "==> Starting Streamlit prototype on port 8501 ..."
        exec streamlit run /app/app.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;
    predict)
        shift
        exec python -m model.predict "$@"
        ;;
    shell)
        exec /bin/bash
        ;;
    *)
        echo "Unknown command: $cmd"
        echo "Usage: docker run insider-threat-detection [train|serve|predict|shell]"
        exit 1
        ;;
esac
