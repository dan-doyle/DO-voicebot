# Use an official Python runtime as the parent image
FROM python:3.11-slim

WORKDIR /app

COPY . .

ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install package dependencies
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Install app specific python libraries
# RUN pip install hypercorn fastapi

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define command to run the app using hypercorn
CMD ["hypercorn", "app:app", "--bind", "0.0.0.0:8080"]