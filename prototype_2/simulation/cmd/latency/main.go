package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/mb"
	"github.com/mochaeng/phoenix-detector/internal/parser"
)

func main() {
	numWorkers := flag.Int("workers", 1, "Number of workers consuming")
	isPublishing := flag.Bool(
		"ispub",
		true,
		"If you want a to publish packets to the [requests_queue]",
	)
	modelPath := flag.String(
		"model",
		"../../../data/fedmedian_model.pt",
		"Path to the PyTorch model",
	)
	csvPath := flag.String(
		"csv",
		"../../../data/10_000-raw-packets.csv",
		"Path to a csv file containing the packets",
	)
	publishInterval := flag.Duration(
		"pub-interval",
		5*time.Millisecond,
		"Interval time for the client publish a packet into [requests_queue]",
	)
	messageLimit := flag.Uint64(
		"msg-limit",
		0,
		"The amount of fixed messages you want to be published. If the value is 0 no limit would be set",
	)
	flag.Parse()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGUSR1, syscall.SIGTERM, os.Interrupt)

	workers, err := mb.GetReadyWorkers(config.AmqpURL, *modelPath, *numWorkers, nil)
	if err != nil {
		log.Fatalf("failed to creatte workers. Error: %v\n", err)
	}
	for _, worker := range workers {
		go worker.ConsumeRequestsQueue()
	}

	messages, err := parser.GetMessages(*csvPath)
	if err != nil {
		log.Panicf("failed to get messages. Error: %v\n", err)
	}

	client, err := mb.GetReadyClient(config.AmqpURL, messages, *messageLimit, *publishInterval)
	if err != nil {
		log.Fatalf("failed to get client. Error: %v\n", err)
	}

	log.Println("Ready to start latencies simulations")

	for {
		select {
		case sig := <-sigChan:
			switch sig {
			case syscall.SIGUSR1:
				if *isPublishing {
					log.Println("Client start to publish messages...")
					go client.StartPublishing()
				}
			case syscall.SIGTERM, os.Interrupt:
				log.Println("Stopping client and workers")
				client.Stop()
				time.Sleep(10 * time.Second)
				for i := 0; i < len(workers); i++ {
					workers[i].Stop()
				}
				return
			}
		}
	}
}
