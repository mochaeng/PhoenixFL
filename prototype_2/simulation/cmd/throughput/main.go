package main

import (
	"flag"
	"log"
	"os"
	"path/filepath"
	"sync"
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
	messageLimit := flag.Uint64(
		"msg-limit",
		0,
		"The amount of fixed messages you want to be published. If the value is 0 not limit would be set",
	)
	modelPath := flag.String(
		"model",
		"../../../data/fedmedian_model.pt",
		"Path to the PyTorch model",
	)
	idleTimeout := flag.String(
		"idle-timeout",
		"",
		"The worker finishes after idle-timeout has been fired. If the value is 0 no timeout would be set.",
	)
	flag.Parse()

	var workerTimeout *time.Duration
	if *idleTimeout != "" {
		duration, err := time.ParseDuration(*idleTimeout)
		if err != nil {
			log.Panicf("could not parser duration time. Error: %v\n", err)
		}
		workerTimeout = &duration
	}

	log.Printf("worker timeout %f\n", workerTimeout.Seconds())

	messages, err := parser.GetMessages(*csvPath)
	if err != nil {
		log.Panicf("failed to get messages. Error: %v\n", err)
	}

	// we don't care about publish interval here.
	// may I find a better way to model this
	client := mb.NewClient(config.AmqpURL, messages, 0, 1*time.Second)
	err = client.Connect()
	if err != nil {
		log.Panicf("Failed to connect: %v\n", err)
	}
	err = client.SetupClient()
	if err != nil {
		log.Panicf("Failed to set up client: %v\n", err)
	}

	log.Printf("client created \n filling the queue with [%d] packets...", *messageLimit)
	var i = uint64(0)
	for i = 0; i < *messageLimit; i++ {
		if err := client.PublishRequestPacket(); err != nil {
			log.Panicf("could not send message. Error: %v\n", err)
		}
	}

	log.Println("Creating workers...")
	workers := make([]*mb.Worker, 0, *numWorkers)
	for i := 0; i < *numWorkers; i++ {
		worker := mb.NewWorker(config.AmqpURL, *modelPath, "../../../data/workers", workerTimeout)
		if err := worker.Connect(); err != nil {
			log.Panicf("could not connect worker to rabbitMQ. Error: %v\n", err)
		}
		if err := worker.SetupWorker(); err != nil {
			log.Panicf("could not setup worker. Error: %v\n", err)
		}
		workers = append(workers, worker)
		log.Printf("worker [%s] created\n", worker.Name)
	}

	// time.Sleep(35 * time.Second)

	log.Println("Starting consuming...")
	var wg sync.WaitGroup
	wg.Add(*numWorkers)
	for _, worker := range workers {
		go func(w *mb.Worker) {
			defer wg.Done()
			w.ConsumeRequestsQueue()
			w.Stop()
		}(worker)
	}
	wg.Wait()

	// aggregates metrics from current simulation number
	dir := "../../../data/workers"
	absPath, err := filepath.Abs(dir)
	if err != nil {
		log.Fatalf("could not resolve absolute path. Error: %v\n", err)
	}

	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		log.Fatalf("directory [%s] does not exist\n", absPath)
	}

	aggregatedThroughput, err := parser.GetAllThroughputsAggregatedRecords(absPath)
	if err != nil {
		log.Fatalf("failed to aggregate throughputs. Error: %v\n", err)
	}

	pathToSave := filepath.Join(dir, "simulations.json")
	if err := parser.CreateJsonFileFromStruct(aggregatedThroughput, pathToSave); err != nil {
		log.Fatalf("failed to create throughput's simulation file. Error: %v\n", err)
	}
}
