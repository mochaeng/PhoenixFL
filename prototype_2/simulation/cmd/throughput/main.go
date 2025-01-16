package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/mb"
	"github.com/mochaeng/phoenix-detector/internal/models"
	"github.com/mochaeng/phoenix-detector/internal/parser"
)

func cleanWorkerIntermediateFiles(root string) error {
	if _, err := os.Stat(root); os.IsNotExist(err) {
		return fmt.Errorf("directory does not exist. Error: %w", err)
	}

	files, err := os.ReadDir(root)
	if err != nil {
		return fmt.Errorf("failed to read directory. Error: %w\n", err)
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		name := file.Name()
		if strings.HasPrefix(name, "worker") && strings.HasSuffix(name, ".json") || strings.HasSuffix(name, ".csv") {
			fullPath := filepath.Join(root, name)
			if err := os.Remove(fullPath); err != nil {
				log.Printf("warning: failed to delete file [%s]. Error: %v\n", name, err)
			}
		}
	}

	return nil
}

func runSingleSimulationRound(messages []*models.ClientRequest, numWorkers int, messageLimit uint64, modelPath string, workerTimeout *time.Duration) error {
	// we don't care about publish interval here.
	// may I find a better way to model this
	client := mb.NewClient(config.AmqpURL, messages, 0, 1*time.Second)
	err := client.Connect()
	if err != nil {
		return fmt.Errorf("failed to connect: %w\n", err)
	}
	err = client.SetupClient()
	if err != nil {
		return fmt.Errorf("failed to set up client: %w\n", err)
	}

	log.Printf("client created \n filling the queue with [%d] packets...", messageLimit)
	var i = uint64(0)
	for i = 0; i < messageLimit; i++ {
		if err := client.PublishRequestPacket(); err != nil {
			return fmt.Errorf("could not send message. Error: %w\n", err)
		}
	}
	client.Stop()

	log.Println("Creating workers...")
	workers := make([]*mb.Worker, 0, numWorkers)
	for i := 0; i < numWorkers; i++ {
		worker := mb.NewWorker(config.AmqpURL, modelPath, "../../../data/workers", workerTimeout)
		if err := worker.Connect(); err != nil {
			return fmt.Errorf("could not connect worker to rabbitMQ. Error: %w\n", err)
		}
		if err := worker.SetupWorker(); err != nil {
			return fmt.Errorf("could not setup worker. Error: %w\n", err)
		}
		workers = append(workers, worker)
		log.Printf("worker [%s] created\n", worker.Name)
	}

	time.Sleep(35 * time.Second)

	log.Println("Starting consuming...")
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	for _, worker := range workers {
		go func(w *mb.Worker) {
			defer wg.Done()
			w.ConsumeRequestsQueue()
			w.Stop()
		}(worker)
	}
	wg.Wait()

	return nil
}

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

	dir := "../../../data/workers"
	absPath, err := filepath.Abs(dir)
	if err != nil {
		log.Fatalf("could not resolve absolute path. Error: %v\n", err)
	}

	for i := 1; i <= 30; i++ {
		if err := runSingleSimulationRound(
			messages,
			*numWorkers,
			*messageLimit,
			*modelPath,
			workerTimeout,
		); err != nil {
			log.Fatalf("could not run simulation. Error: %v", err)
		}

		if _, err := os.Stat(absPath); os.IsNotExist(err) {
			log.Fatalf("directory [%s] does not exist\n", absPath)
		}

		aggregatedThroughput, err := parser.AggregateWorkerThroughputData(absPath)
		if err != nil {
			log.Fatalf("failed to aggregate throughputs. Error: %v\n", err)
		}

		fileName := fmt.Sprintf("%d-%d-simulation-throughput.json", *numWorkers, i)
		pathToSave := filepath.Join(dir, fileName)
		if err := parser.CreateJsonFileFromStruct(aggregatedThroughput, pathToSave); err != nil {
			log.Fatalf("failed to create throughput's simulation file. Error: %v\n", err)
		}

		if err := cleanWorkerIntermediateFiles(absPath); err != nil {
			log.Printf("warning: could not delete intermediate files, %s)\n", err)
		}

	}

	simulationResult, err := parser.GetThroughputSimulationResult(absPath, *numWorkers)
	if err != nil {
		log.Fatalf("failed to get throughput simulation's results. Error: %s\n", err)
	}

	fileName := fmt.Sprintf("simulation-%d-workers.json", *numWorkers)
	pathToSave := filepath.Join(dir, fileName)
	if err := parser.CreateJsonFileFromStruct(simulationResult, pathToSave); err != nil {
		log.Fatalf("failed to create simulation result file. Error: %v\n", err)
	}

}
