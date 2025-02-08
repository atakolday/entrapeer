# Use official Python image
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Run the main application
CMD ["python", "main.py"]