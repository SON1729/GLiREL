FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-model.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-model.txt fastapi==0.115.12 uvicorn==0.34.2
RUN useradd -m -u 1000 user
COPY --chown=user:user core.py inference.py service.py ./
USER user
ENV HF_HOME=/home/user/.cache/huggingface TOKENIZERS_PARALLELISM=false GLIREL_DEVICE=cpu
EXPOSE 7860
CMD ["uvicorn","service:app","--host","0.0.0.0","--port","7860","--workers","1"]
