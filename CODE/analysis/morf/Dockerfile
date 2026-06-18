# DRAFT Docker image for the MORF job (built by us, run inside the MORF enclave).
# MORF runs OUR container, so we control the base image — use a modern Python + torch/transformers
# (MORF's "Python 3.6" recommendation is for the morf-api *client*, not the container internals).
#
# Build + push to a public registry, then point config.properties at the image:
#   docker build -t <registry>/morf-dropout-text:latest .
#   docker push   <registry>/morf-dropout-text:latest
#
# Note: bake the DistilBERT weights into the image (the enclave may have no outbound internet):
#   the snippet below pre-downloads distilbert-base-uncased at build time.

FROM python:3.11-slim

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers scikit-learn pandas numpy

# Pre-cache the frozen encoder so the container needs no network at run time.
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
    AutoModel.from_pretrained('distilbert-base-uncased')"

WORKDIR /app
COPY mwe_dropout_text.py /app/mwe.py

# MORF invokes: docker run <image> --mode {extract|extract-holdout|train|test}
ENTRYPOINT ["python3", "mwe.py"]
