APP_DIR="/home/ubuntu/ml_project"
LOG_FILE="$APP_DIR/app_output.log"

# Check for existing Flask Application Process
echo "Checking for existing Flask Application Process"
EXISTING_PID=$(pgrep -f "python3 app.py")


if [-n "${EXISTING_PID}" ]; then
    echo "Killing existing Flask Application Process"
    kill -9 $EXISTING_PID
else:
    echo "No existing Flask Application Process found"
fi


#navigate to app dir
cd $APP_DIR

#set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

#install dependencies
echo "Installing dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

#run the app
nohup python3 app.py > $LOG_FILE 2>&1 &

NEW_PID=$!

echo "Flask application started with PID $NEW_PID