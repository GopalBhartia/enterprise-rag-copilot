FROM python:3.11-slim

WORKDIR /app

# system deps (important for some ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install dependencies first (better caching)
COPY pyproject.toml uv.lock* /app/

# install uv
RUN pip install uv

# install dependencies via uv
RUN uv pip install --system .

# copy full project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]