#!/bin/bash
set -e

chmod +x setup.sh
./setup.sh    # Set up skin_env and install packages

sleep 5

echo "Activated VENV: $VIRTUAL_ENV"


# Run Flask app
echo "Starting Flask chatbot..."
flask run -p 5001 --debug
