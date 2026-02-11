FROM python:3.11-slim
WORKDIR /insurance_app
COPY requirements.txt .
RUN pip install --no-cache-dir -r  requirements.txt
COPY . . 
EXPOSE 8000
CMD ["uvicorn","insurance_app:app", "--host", "0.0.0.0", "--port", "8000"]