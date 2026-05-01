FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY llm_tester ./llm_tester
RUN pip install --no-cache-dir .

ENTRYPOINT ["llm-tester"]
