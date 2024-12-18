package main

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	amqp "github.com/rabbitmq/amqp091-go"
)

type Metadata struct {
	SourceIpAddr  string `json:"IPV4_SRC_ADDR"`
	SourcePortNum int    `json:"L4_SRC_PORT"`
	DestIpAddr    string `json:"IPV4_DST_ADDR"`
	DestPortNum   int    `json:"L4_DST_PORT"`
}

type ClassifiedPacket struct {
	Metadata           Metadata `json:"metadata"`
	ClassificationTime float64  `json:"classification_time"`
	TotalTime          float64  `json:"total_time"`
	WorkerName         string   `json:"worker_name"`
	IsMalicious        bool     `json:"is_malicious"`
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

func webSocketHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("websocket upgrade failed. Error: %s", err)
		return
	}
	defer conn.Close()

	log.Println("Client connected")

	for {
		err = conn.WriteMessage(websocket.BinaryMessage, []byte("aaa"))
		if err != nil {
			log.Printf("failed to write message. Error: %s", err)
			break
		}
		time.Sleep(2 * time.Second)
	}
}

func startRabbitMQConsumer(packetChan chan ClassifiedPacket) {
	conn, err := amqp.Dial("amqp://guest:guest@localhost:5672/")
	if err != nil {
		log.Panicf("failed to connect to rabbitMQ. Error: %s", err)
	}
	defer conn.Close()

	ch, err := conn.Channel()
	if err != nil {
		log.Panicf("failed to open a channel. Error: %s", err)
	}
	defer ch.Close()

	err = ch.ExchangeDeclare(
		"packets", // name
		"direct",  // type
		true,      // durable
		false,     // auto-deleted
		false,     // internal
		false,     // no-wait
		nil,       // arguments
	)
	if err != nil {
		log.Panicf("failed to declare an exchange. Error: %s", err)
	}

	q, err := ch.QueueDeclare(
		"alerts_queue", // name
		true,           // durable
		false,          // delete when unused
		false,          // exclusive
		false,          // no-wait
		nil,            // arguments
	)
	if err != nil {
		log.Panicf("failed to open declare a queue. Error: %s", err)
	}

	err = ch.QueueBind(
		q.Name,    // queue name
		q.Name,    // routing key
		"packets", // exchange
		false,
		nil,
	)
	if err != nil {
		log.Printf("failed to bind a queue. Error: %s", err)
	}

	err = ch.Qos(2, 0, false)
	if err != nil {
		log.Panicf("failed to set QoS. Error: %s", err)
	}

	msgs, err := ch.Consume(
		q.Name, // queue
		"",     // consumer
		false,  // auto-ack
		false,  // exclusive
		false,  // no-local
		false,  // no-wait
		nil,    // args
	)
	if err != nil {
		log.Panicf("failed to register a consumer. Error: %s", err)
	}

	for d := range msgs {
		log.Printf("Received a message: %s", d.Body)

		var packet ClassifiedPacket
		err := json.Unmarshal([]byte(d.Body), &packet)
		if err != nil {
			log.Printf("Error parsing json: %s\n", err)
			d.Nack(false, true)
			continue
		}
		packetChan <- packet
		d.Ack(false)
	}

	log.Printf(" [*] Waiting for messages. To exit press CTRL+C")
	select {}
}

func main() {
	packetChan := make(chan ClassifiedPacket)
	var clients sync.Map
	go startRabbitMQConsumer(packetChan)

	http.HandleFunc("/live-classifications", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("failed to upgrade websocket. Error: %s\n", err)
			return
		}
		log.Println("Client connected")

		clientID := conn.RemoteAddr().String()
		clients.Store(clientID, conn)

		defer func() {
			conn.Close()
			clients.Delete(clientID)
			log.Printf("Client disconnected id [%s]\n", clientID)
		}()

		// for packet := range packetChan {
		// 	err := conn.WriteJSON(packet)
		// 	if err != nil {
		// 		log.Printf("could not send message to client. Error: %s\n", err)
		// 		break
		// 	}
		// }
	})

	log.Println("server started on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
