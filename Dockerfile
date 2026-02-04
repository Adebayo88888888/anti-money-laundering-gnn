# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first (to cache dependencies)
COPY requirements.txt .

# 4. Install dependencies (no cache to save space)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Command to run the application
# We use "production.app:app" because your file is inside the 'production' folder
CMD ["uvicorn", "production.app:app", "--host", "0.0.0.0", "--port", "8000"]