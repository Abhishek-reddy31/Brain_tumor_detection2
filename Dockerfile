# Use a base image with Python
FROM python:3.11.1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install Flask keras==2.14.0 numpy pillow tensorflow==2.14.0 ml-dtypes==0.2.0 tensorboard==2.14.1 gunicorn

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the Flask application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
