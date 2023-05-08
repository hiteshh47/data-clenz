FROM python:3.9-slim-buster
LABEL owner="Hitesh A"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "ISI-app.py"]

