# Combined image for Hugging Face Spaces (Docker SDK): one container running both the
# FastAPI backend and the Streamlit UI together, since Spaces exposes exactly one public
# port per Space and, unlike Render, has no card-free way to run two linked services.
# FastAPI listens on 127.0.0.1 only (internal, reached by Streamlit inside the same
# container); Streamlit is the one process actually exposed on the Space's public port.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ui.txt

# Pre-download the MLIP at build time, same reasoning as Dockerfile.api: keeps a live
# Hugging Face Hub call off the critical path of the Space's first real request.
RUN python -c "import matgl; matgl.load_model('M3GNet-PES-MatPES-PBE-2025.2')"

COPY agent/ agent/
COPY api/ api/
COPY observability/ observability/
COPY ui/ ui/
COPY start_hf_space.sh .
RUN chmod +x start_hf_space.sh

# Hugging Face Spaces (Docker SDK) always routes external traffic to port 7860.
EXPOSE 7860
CMD ["./start_hf_space.sh"]
