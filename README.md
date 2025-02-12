<div align="center">
  <img src="./icon.png" alt="Caliburn logo."/>
  <br />
  <br/>
  <a href="https://www.flaticon.com/free-icons/phoenix" title="phoenix icon" target="_blank">art from Flaticon</a>
</div>

# PhoenixFL

PhoenixFL is an asynchronous and high-performance solution for classifying network packets in heterogeneous IoT environments using neural networks. It leverages federated learning to train models while preserving participant data privacy. Additionally, it includes a web interface for real-time visualization of system classifications.

## Web Interface Demo

https://github.com/user-attachments/assets/a53c4300-abc8-4853-9e3d-6e5c47cb9cd2

## Design

![Phoenix Design](/imgs/phoenix_complete.png)

Initially, a device captures a packet sent to it and constructs a request message containing the packet information, sender details, and some metadata. The client then sends this message to the request exchange, which forwards it to the appropriate queue. One of the Worker subscribed to this queue eventually consumes the message and classifies the packet as malicious or not. After this step, the worker publishes the result to the alert exchange, which in turn redirects the message to the alert queue. The backend, subscribed to this queue, is notified of the classification and processes the alert message, forwarding the results to the frontend via WebSocket. Finally, the frontend displays the information to the network administrator, allowing them to assess the network's security status and take appropriate action.

## Structure

- **prototype_01**: This directory contains the code used to train a model for network packet classification using the Federated Learning technique.

- **prototype_02**: This directory contains the code for the RabbitMQ queue. It includes the simulation of devices sending packets to the queue, along with workers consuming the packets using a federated model trained in the previous prototype.

- **prototype_03**: This directory contains the server that consumes messages from the alert queue and sends the information to the frontend via WebSocket.

## Running

To run the code, you should first navigate to the `prototype_2` directory:

```sh
cd prototype_2
```

First, you need to build a shared object with the LibTorch and CUDA dependencies so that Golang can use it. [Go to this folder](/prototype_2/simulation/internal/torchbidings/) for more details on what needs to be done.

Next, inside the root of the `prototype_02` folder, start RabbitMQ:

```sh
docker compose up --build
```

Then, you need to run a simulation. There is a Makefile with three modes. To execute this project, you can use the simplest one:

```sh
make run-normalflow PUB_INTERVAL=1s WORKERS=2
```

The `PUB_INTERVAL` variable will publish a packet every 1 second, simulating a device sending data to the message broker. You can use other values like 500ms to send a packet every 500 milliseconds, etc. You can also configure the number of `WORKERS` as needed.

After that, to run the web interface, navigate to the prototype_03 directory. It is divided into backend and frontend. Inside the backend folder, run:

```sh
go run cmd/main.go
```

Now, inside the frontend folder, you can start the interface with:

```sh
pnpm dev
```

## Results

With 5 workers, the system achieved a classification throughput of 4,348 p/s, taking an average of 230 µs to classify a single packet.
