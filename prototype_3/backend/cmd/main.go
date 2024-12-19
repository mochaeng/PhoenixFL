package main

import (
	"log"
	"net/http"

	"github.com/mochaeng/phoenixfl/internal/consumer"
	"github.com/mochaeng/phoenixfl/internal/handler"
	"github.com/mochaeng/phoenixfl/internal/hub"
	"github.com/mochaeng/phoenixfl/internal/models"
)

func StartWebSocketServer(packetsChan <-chan models.ClassifiedPacketResponse) {
	hub := hub.NewClientsHub()

	go func() {
		for packet := range packetsChan {
			hub.BroadcastPacket(packet)
		}
	}()

	http.HandleFunc("/live-classifications", handler.LiveClassificationsHandler(hub))
	log.Println("server started on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func main() {
	packetsChan := make(chan models.ClassifiedPacketResponse)

	conn, ch := consumer.ConnectToRabbitMQ()
	defer conn.Close()
	defer ch.Close()

	alertsQueue, err := consumer.GetAlertsQueue(ch)
	if err != nil {
		log.Panicf("Could not get alerts queue. Error: %s\n", err)
	}

	go consumer.ConsumeAlertMessages(ch, alertsQueue, packetsChan)
	StartWebSocketServer(packetsChan)
}
