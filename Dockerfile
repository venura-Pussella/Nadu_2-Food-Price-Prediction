# Use TensorFlow base image with Python 3.11
FROM tensorflow/tensorflow:2.18.0

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 80

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]