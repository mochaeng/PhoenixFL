package mq

import (
	"encoding/json"
	"log"

	"github.com/mochaeng/phoenixfl/internal/algorithms"
	"github.com/mochaeng/phoenixfl/internal/models"
	amqp "github.com/rabbitmq/amqp091-go"
)

func ConnectToRabbitMQ() (*amqp.Connection, *amqp.Channel) {
	conn, err := amqp.Dial("amqp://guest:guest@localhost:5672/")
	if err != nil {
		log.Panicf("failed to connect to rabbitMQ. Error: %s", err)
	}

	ch, err := conn.Channel()
	if err != nil {
		log.Panicf("failed to open a channel. Error: %s", err)
	}

	return conn, ch
}

func GetAlertsQueue(ch *amqp.Channel) (*amqp.Queue, error) {
	err := ch.ExchangeDeclare("packets", // name
		"direct", // type
		true,     // durable
		false,    // auto-deleted
		false,    // internal
		false,    // no-wait
		nil,      // arguments
	)
	if err != nil {
		return nil, err
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
		return nil, err
	}

	err = ch.QueueBind(
		q.Name,    // queue name
		q.Name,    // routing key
		"packets", // exchange
		false,
		nil,
	)
	if err != nil {
		return nil, err
	}

	err = ch.Qos(2, 0, false)
	if err != nil {
		return nil, err
	}

	return &q, nil
}

func ConsumeAlertsMessages(ch *amqp.Channel, queue *amqp.Queue, packetsChan chan models.PacketWithStatsResponse) error {
	msgs, err := ch.Consume(
		queue.Name, // queue
		"",         // consumer
		false,      // auto-ack
		false,      // exclusive
		false,      // no-local
		false,      // no-wait
		nil,        // args
	)
	if err != nil {
		return err
	}

	stats := algorithms.NewPacketStats()
	for d := range msgs {
		var packet models.Packet
		err := json.Unmarshal([]byte(d.Body), &packet)
		if err != nil {
			log.Printf("Error parsing json: %s\n", err)
			d.Nack(false, true)
			continue
		}

		stats.Update(packet)

		packetResponse := models.PacketWithStatsResponse{}
		packetResponse.Packet = &packet
		packetResponse.Stats = models.StatsResponse{
			TotalPackets:   stats.TotalPackets,
			TotalMalicious: stats.TotalMalicious,
		}
		malicious := stats.GetTopMaliciousIps(5)
		log.Println(len(malicious))
		for i := 0; i < len(malicious); i++ {
			log.Printf("%+v ", malicious[i])
		}

		// log.Printf("Received a message: %+v\n", packetResponse)
		packetsChan <- packetResponse
		d.Ack(false)
	}

	log.Printf(" [*] Waiting for messages. To exit press CTRL+C")
	return nil
}
