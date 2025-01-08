package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/mb"
	"github.com/mochaeng/phoenix-detector/internal/parser"
)

func main() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGUSR1, syscall.SIGTERM, os.Interrupt)

	numWorkers := 1
	workers := make([]*mb.Worker, 0, numWorkers)
	for i := 0; i < numWorkers; i++ {
		worker := mb.NewWorker(config.AmqpURL, config.ModelPath)
		if err := worker.Connect(); err != nil {
			log.Panicf("could not connect worker to rabbitMQ. Error: %v\n", err)
		}
		if err := worker.SetupWorker(); err != nil {
			log.Panicf("could not setup rabbitMQ. Error: %v\n", err)
		}
		go worker.ConsumeRequestsRequeue()
		workers = append(workers, worker)
	}

	columnsToRemove := []string{
		"IPV4_SRC_ADDR",
		"IPV4_DST_ADDR",
		"L4_SRC_PORT",
		"L4_DST_PORT",
	}
	messages, err := parser.ParseCSV("../../data/10_000-raw-packets.csv", columnsToRemove)
	if err != nil {
		log.Panicf("failed to parse csv packets. Error: %v\n", err)
	}

	client := mb.NewClient(config.AmqpURL, messages)
	err = client.Connect()
	if err != nil {
		log.Panicf("Failed to connect: %v\n", err)
	}
	err = client.SetupClient()
	if err != nil {
		log.Panicf("Failed to set up client: %v\n", err)
	}

	log.Println("Client and Workers are ready")

	for {
		select {
		case sig := <-sigChan:
			switch sig {
			case syscall.SIGUSR1:
				log.Println("Client start to publish messages...")
				// go client.StartPublishing()
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
