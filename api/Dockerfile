FROM python:3.10-slim

WORKDIR /app

# Copy API requirements explicitly
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code and local models
COPY api/ /app/api/
COPY models/ /app/models/

EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
