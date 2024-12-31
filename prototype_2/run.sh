#!/bin/bash

rm -f data/workers/worker*

source .venv/bin/activate

NUM_WORKERS=0
NUM_CLIENTS=1

echo "Starting Workers..."
WORKER_PIDS=()
for i in $(seq 1 $NUM_WORKERS); do
    poetry run python -m rpc.worker &
    WORKER_PIDS+=($!)
done

echo "Starting Clients..."
CLIENT_PIDS=()
for i in $(seq 1 $NUM_CLIENTS); do
    poetry run python -m rpc.client &
    CLIENT_PIDS+=($!)
done

# clients sending packages for T seconds
sleep 60
# Waiting for clients to send packets
# for pid in "${CLIENT_PIDS[@]}"; do
#     wait $pid
# done
for pid in "${CLIENT_PIDS[@]}"; do
    kill -SIGINT $pid
done
echo "Clients finished processing."

# # waiting for workers to finish consuming packets from the queue
sleep 30

echo "Killing workers"
# for pid in "${WORKER_PIDS[@]}"; do
#     wait $pid
# done
for pid in "${WORKER_PIDS[@]}"; do
    kill -SIGINT ${pid}
done
