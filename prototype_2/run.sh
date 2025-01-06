#!/bin/bash

rm -f data/workers/worker*

echo "Starting prototype 02 script..."
cd golang/cmd
go build *.go
./main &
mb_pid=$!
cd ../..

# preparation time
sleep 5

# start client packets sending for T seconds
kill -SIGUSR1 ${mb_pid}
sleep 60

# stop script
kill -SIGTERM ${mb_pid}
echo "prototype 02 script finished."
