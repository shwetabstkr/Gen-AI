FROM python:3.11-slim
 
WORKDIR /app
 
# Install system dependencies
RUN apt-get update && apt-get install -y build-essential git
 
# Copy requirements first for better build caching
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy application code
COPY . .
 
EXPOSE 8501
 
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
 