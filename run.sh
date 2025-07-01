#!/bin/bash
set -e

chmod +x setup.sh
./setup.sh    # Set up skin_env and install packages

source skin_env/bin/activate # activate venv

# Run Flask app
echo "Starting Flask chatbot..."
flask run -p 5001
