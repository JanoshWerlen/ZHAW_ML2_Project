# Use the official Python image from the Docker Hub
FROM python:3.12.1-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements_minimal.txt requirements_minimal.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Copy the rest of the application code
COPY . .

# List the contents of the data directory to debug file copy issues
RUN ls -l /app/App/data/3_embedded/ABPR

# Expose the port that your application runs on
EXPOSE 5001

# Specify the command to run on container start
CMD ["python", "App/app.py"]
