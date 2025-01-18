#!/bin/bash

# Application directory
APP_DIR="/home/ubuntu/ml_project"
LOG_FILE="$APP_DIR/app_output.log"

# Check for existing Flask application processes
echo "Checking for existing Flask application processes..."
EXISTING_PID=$(pgrep -f "python3 app.py")

if [ -n "$EXISTING_PID" ]; then
    echo "Found running Flask application process with PID: $EXISTING_PID. Terminating it..."
    kill -9 "$EXISTING_PID"
    echo "Terminated previous Flask application process."
else
    echo "No existing Flask application processes found."
fi

# Navigate to the application directory
cd "$APP_DIR"

# Set up virtual environment
echo "Setting up the virtual environment..."

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Start the Flask application with nohup
echo "Starting Flask application..."
nohup python3 app.py > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "Flask application started with PID: $NEW_PID"
