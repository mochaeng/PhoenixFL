#!/bin/bash

source .venv/bin/activate

echo "Starting Workers..."
poetry run python -m rpc.worker &
WORKER_PID=$!

echo "Starting Clients..."
poetry run python -m rpc.client &
CLIENT_PID=$!
poetry run python -m rpc.client &
CLIENT_PID1=$!
poetry run python -m rpc.client &
CLIENT_PID2=$!
sleep 60
kill -SIGINT $CLIENT_PID
kill -SIGINT $CLIENT_PID1
kill -SIGINT $CLIENT_PID2
echo "Client finished processing."

sleep 60
kill -SIGINT $WORKER_PID
# wait $WORKER_PID
