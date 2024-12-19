# Prototype 03

- The backend, implemented in Go, consumes classified alerts from a RabbitMQ queue (`alerts_queue`) and broadcasts them to connected clients via WebSocket. 

- The frontend provides a user interface to display these alerts in real time.
