FROM python:3.10-slim
WORKDIR /app
COPY mlflow_comparator.py .
RUN pip install --no-cache-dir mlflow pandas matplotlib seaborn
CMD ["python", "mlflow_comparator.py"]
