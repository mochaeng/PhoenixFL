# Prototype 02

This prototype aims to evaluate a distributed architecture for a classification system.

# Running

- **RabbitMQ** instance running.

- run a single client:

```sh
poetry run python -m rpc.client
```

- run a single worker:

```sh
poetry run python -m rpc.worker
```
