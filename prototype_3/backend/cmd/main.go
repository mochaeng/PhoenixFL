package main

import (
	"log"

	"github.com/mochaeng/phoenixfl/internal/models"
	"github.com/mochaeng/phoenixfl/internal/mq"
	"github.com/mochaeng/phoenixfl/internal/realtime"
)

func main() {
	packetsChan := make(chan models.ClassifiedPacketResponse)

	conn, ch := mq.ConnectToRabbitMQ()
	defer conn.Close()
	defer ch.Close()

	alertsQueue, err := mq.GetAlertsQueue(ch)
	if err != nil {
		log.Panicf("Could not get alerts queue. Error: %s\n", err)
	}

	go mq.ConsumeAlertsMessages(ch, alertsQueue, packetsChan)
	realtime.StartWebSocketServer(packetsChan)
}
