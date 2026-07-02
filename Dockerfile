FROM python:3.10-slim as builder
WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
RUN pip install -e .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "src/agentic_profile_matching/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
