FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY evals/ evals/
COPY sample_logs/ sample_logs/
COPY tests/ tests/
ENV PYTHONPATH=/app/src
RUN python evals/run_evals.py
EXPOSE 8000
CMD ["uvicorn", "ci_triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
