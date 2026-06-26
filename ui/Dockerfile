FROM python:3.11-slim

WORKDIR /ui

RUN pip install streamlit requests

COPY ui /ui

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]