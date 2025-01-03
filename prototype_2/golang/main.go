package main

import (
	"context"
	"log"
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

	messages, err := parser.ParseCSV("../data/10_000-raw-packets.csv", columnsToRemove)
	if err != nil {
		log.Panicf("failed to parse csv packets. Error: %v\n", err)
	}

	client := mb.NewClient("amqp://guest:guest@localhost:5672/", messages)

	err = client.Connect()
	if err != nil {
		log.Panicf("Failed to connect: %v\n", err)
	}

	err = client.SetupChannel()
	if err != nil {
		log.Panicf("Failed to set up channel: %v\n", err)
	}

	err = client.SetupPublisherConfirms()
	if err != nil {
		log.Panicf("Failed to set up publisher confirms: %v\n", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go client.StartPublishing(ctx)
	<-ctx.Done()
	client.Stop()
}
