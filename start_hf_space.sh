#!/bin/bash
# Starts FastAPI in the background on localhost only, waits until it actually answers
# (not a fixed sleep, since a fixed delay either wastes time or races a slow cold start),
# then runs Streamlit in the foreground as the process Hugging Face Spaces keeps alive.
set -e

export API_BASE_URL="http://127.0.0.1:8000"

uvicorn api.main:app --host 127.0.0.1 --port 8000 &

python3 -c "
import time, urllib.request
for _ in range(60):
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit('FastAPI did not become healthy in time')
"

exec streamlit run ui/app.py --server.address=0.0.0.0 --server.port="${PORT:-7860}"
