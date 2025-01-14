package main

import (
	"flag"
	"log"
	"time"

	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/mb"
	"github.com/mochaeng/phoenix-detector/internal/parser"
)

func main() {
	csvPath := flag.String("csv", "../../../data/10_000-raw-packets.csv", "Path to a csv file containing the packets")
	publishInterval := flag.Duration("pub-interval", 5*time.Millisecond, "Interval time in wich the client will publish a packet into [requests_queue]")
	messageLimit := flag.Uint64("msg-limit", 0, "The amount of fixed messages you want to be published. If the value is 0 not limit would be set")

	// numWorkers := flag.Int("workers", 1, "Number of workers consuming")
	// // isPublishing := flag.Bool("ispub", true, "If you want a to publish packets to the [requests_queue]")
	// modelPath := flag.String("model", "../../../data/fedmedian_model.pt", "Path to the PyTorch model")
	// csvPath := flag.String("csv", "../../../data/10_000-raw-packets.csv", "Path to a csv file containing the packets")
	// publishInterval := flag.Duration("pub-interval", 5*time.Millisecond, "Interval time in wich the client will publish a packet into [requests_queue]")
	// messageLimit := flag.Uint64("msg-limit", 0, "The amount of fixed messages you want to be published. If the value is 0 not limit would be set")
	// flag.Parse()

	// columnsToRemove := []string{
	// 	"IPV4_SRC_ADDR",
	// 	"IPV4_DST_ADDR",
	// 	"L4_SRC_PORT",
	// 	"L4_DST_PORT",
	// }
	// messages, err := parser.ParsePacketsCSV(*csvPath, columnsToRemove)
	// if err != nil {
	// 	log.Panicf("failed to parse csv packets. Error: %v\n", err)
	// }

	// client := mb.NewClient(config.AmqpURL, messages, *messageLimit, *publishInterval)
	// err = client.Connect()
	flag.Parse()

	messages, err := parser.GetMessages(*csvPath)
	if err != nil {
		log.Panicf("failed to get messages. Error: %v\n", err)
	}

	client := mb.NewClient(config.AmqpURL, messages, 0, *publishInterval)
	err = client.Connect()
	if err != nil {
		log.Panicf("Failed to connect: %v\n", err)
	}
	err = client.SetupClient()
	if err != nil {
		log.Panicf("Failed to set up client: %v\n", err)
	}

	var i = uint64(0)
	for i = 0; i < *messageLimit; i++ {
		if err := client.PublishRequestPacket(); err != nil {
			log.Panicf("could not send message. Error: %v\n", err)
		}
	}

}
