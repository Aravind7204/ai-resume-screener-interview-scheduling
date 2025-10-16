# Use a stable, officially supported Python image
FROM python:3.10.14-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure necessary NLTK data is downloaded if used (adjust if not needed or already downloaded)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create the uploads and instance/temp directories needed by the app
RUN mkdir -p uploads
RUN mkdir -p instance/temp

# Expose the port your Flask app runs on (Gunicorn typically uses 8000 by default)
EXPOSE 8000

# Command to run your application using Gunicorn
# Ensure FLASK_DEBUG is set to False in Render environment variables for production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "run:app"]
