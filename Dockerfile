FROM python:3.11-slim

# Install system dependencies (MariaDB database and build tools)
RUN apt-get update && apt-get install -y \
    mariadb-server \
    mariadb-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirement file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Copy and set executable permissions for the startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Run startup script
CMD ["/start.sh"]
