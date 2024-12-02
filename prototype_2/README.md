# Prototype 02

This prototype aims to evaluate a distributed architecture for a classification system.

# Running

- start rabbitMQ:

```sh
docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.13-management
```

- run a single worker:

```sh
poetry run python -m rpc.rpc_client
```
