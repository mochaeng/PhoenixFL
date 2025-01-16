package normalflow

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
	csvPath := flag.String(
		"csv",
		"../../../data/10_000-raw-packets.csv",
		"Path to a csv file containing the packets",
	)
	modelPath := flag.String(
		"model",
		"../../../data/fedmedian_model.pt",
		"Path to the PyTorch model",
	)
	publishInterval := flag.Duration(
		"pub-interval",
		5*time.Millisecond,
		"Interval time for the client publish a packet into [requests_queue]",
	)
	flag.Parse()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGTERM, os.Interrupt)

	messages, err := parser.GetMessages(*csvPath)
	if err != nil {
		log.Panicf("failed to get messages. Error: %v\n", err)
	}

	client, err := mb.GetReadyClient(config.AmqpURL, messages, 0, *publishInterval)
	if err != nil {
		log.Fatalf("could not create client. Error: %v\n", err)
	}
	// client := mb.NewClient(config.AmqpURL, messages, 0, *publishInterval)
	// err = client.Connect()
	// if err != nil {
	// 	log.Panicf("Failed to connect: %v\n", err)
	// }
	// err = client.SetupClient()
	// if err != nil {
	// 	log.Panicf("Failed to set up client: %v\n", err)
	// }
	go client.StartPublishing()

	workers, err := mb.GetReadyWorkers(config.AmqpURL, *modelPath, *numWorkers, nil)
	if err != nil {
		log.Fatalf("could not create workers. Error: %v\n", err)
	}
	for _, worker := range workers {
		go worker.ConsumeRequestsQueue()
	}

	// workers := make([]*mb.Worker, 0, *numWorkers)
	// for i := 0; i < *numWorkers; i++ {
	// 	worker := mb.NewWorker(config.AmqpURL, *modelPath, "../../../data/workers", nil)
	// 	if err := worker.Connect(); err != nil {
	// 		log.Panicf("could not connect worker to rabbitMQ. Error: %v\n", err)
	// 	}
	// 	if err := worker.SetupWorker(); err != nil {
	// 		log.Panicf("could not setup worker. Error: %v\n", err)
	// 	}
	// 	go worker.ConsumeRequestsQueue()
	// 	workers = append(workers, worker)
	// }

	for {
		select {
		case sig := <-sigChan:
			switch sig {
			case syscall.SIGTERM, os.Interrupt:
				log.Println("Stopping client and workers")
				client.Stop()
				for i := 0; i < len(workers); i++ {
					workers[i].Stop()
				}
				return
			}
		}
	}

}
