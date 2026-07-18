#!/bin/bash

# Initialize MariaDB data directory if not done
if [ ! -d "/var/lib/mysql/mysql" ]; then
    echo "[INFO] Initializing MariaDB data directory..."
    mysql_install_db --user=mysql --datadir=/var/lib/mysql
fi

# Start MariaDB service in the background
echo "[INFO] Starting MariaDB service..."
mysqld_safe --user=mysql --datadir=/var/lib/mysql &

# Wait for MariaDB to start up
echo "[INFO] Waiting for MariaDB to start..."
until mysqladmin ping --silent; do
    sleep 1
done

echo "[INFO] MariaDB is up and running."

# Configure credentials matching the project code
mysql -u root -e "ALTER USER 'root'@'localhost' IDENTIFIED BY 'Vinoth@0202'; FLUSH PRIVILEGES;"
mysql -u root -p"Vinoth@0202" -e "CREATE DATABASE IF NOT EXISTS business_insite;"

# Set environment variables for DB connection
export DB_HOST="127.0.0.1"
export DB_USER="root"
export DB_PASSWORD="Vinoth@0202"
export DB_NAME="business_insite"
export DB_PORT=3306

# Run ETL database setup and mock data seeding
echo "[INFO] Running db_setup.py..."
python db_setup.py

# Start Streamlit application
echo "[INFO] Starting Streamlit..."
streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
