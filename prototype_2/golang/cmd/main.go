package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/mochaeng/phoenix-detector/internal/mb"
	"github.com/mochaeng/phoenix-detector/internal/parser"
)

func main() {
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

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGUSR1, syscall.SIGTERM)

	client := mb.NewClient("amqp://guest:guest@localhost:5672/", messages)
	err = client.Connect()
	if err != nil {
		log.Panicf("Failed to connect: %v\n", err)
	}
	err = client.SetupRabbitMQ()
	if err != nil {
		log.Panicf("Failed to set up channel: %v\n", err)
	}
	err = client.SetupPublisherConfirms()
	if err != nil {
		log.Panicf("Failed to set up publisher confirms: %v\n", err)
	}

	for {
		sig := <-sigChan
		switch sig {
		case syscall.SIGUSR1:
			log.Println("Ready to start publishing")
			go client.StartPublishing()
		case syscall.SIGTERM:
			log.Println("Stopping client")
			client.Stop()
			return
		}
	}

	// go client.StartPublishing()
	// <-ctx.Done()
	// client.Stop()
}
