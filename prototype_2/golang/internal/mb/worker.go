package mb

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/google/uuid"
	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/models"
	amqp "github.com/rabbitmq/amqp091-go"
)

type Worker struct {
	name          string
	amqpURL       string
	conn          *amqp.Connection
	channel       *amqp.Channel
	latencies     []float64
	processedPkgs int

	stopConsume   chan bool
	requestsQueue *amqp.Queue
	requestsMsgs  <-chan amqp.Delivery
}

func NewWorker(amqpURL string) *Worker {
	return &Worker{
		name:    fmt.Sprintf("worker_%s", uuid.NewString()),
		amqpURL: amqpURL,
	}
}

func (w *Worker) Connect() error {
	conn, err := amqp.Dial(w.amqpURL)
	if err != nil {
		return err
	}
	w.conn = conn

	channel, err := w.conn.Channel()
	if err != nil {
		return err
	}
	w.channel = channel

	return nil
}

func (w *Worker) SetupRabbitMQ() error {
	if w.channel == nil {
		return config.ErrInvalidChannel
	}

	err := SetQoS(w.channel, 1)
	if err != nil {
		return err
	}

	err = PacketExchangeDeclare(w.channel)
	if err != nil {
		return err
	}

	requestsQueue, err := GetRequestsQueue(w.channel)
	if err != nil {
		return err
	}

	_, err = GetAlertsQueue(w.channel)
	if err != nil {
		return err
	}

	w.requestsQueue = requestsQueue

	return nil
}

func (w *Worker) ConsumeRequestsRequeue() {
	for {
		select {
		case <-w.stopConsume:
			log.Printf("worker %s has stopped\n", w.name)
		case delivery := <-w.requestsMsgs:
			var msg models.ClientRequest
			err := json.Unmarshal([]byte(delivery.Body), &msg)
			if err != nil {
				log.Printf("error parsing message. Error: %v\n", err)
				delivery.Nack(false, true)
				continue
			}
			log.Printf("%v+\n", msg.Metadata)
			delivery.Ack(false)
		}
	}
}

func (w *Worker) SetRequestsQueueMessages() error {
	msgs, err := w.channel.Consume(
		config.RequestsQueueName,
		"",    // consumer
		false, // auto-ack
		false, // exclusive
		false, // no-local
		false, // no-wait
		nil,   // args
	)
	if err != nil {
		return fmt.Errorf("could not consume from queue [requests_queue]. Error: %v\n", err)
	}
	w.requestsMsgs = msgs
	return nil
}

func (w *Worker) Stop() {
	if w == nil {
		return
	}

	log.Printf("Stopping worker %s...\n", w.name)
	w.stopConsume <- true
	if w.channel != nil {
		if err := w.channel.Close(); err != nil {
			log.Printf("could not close channel. Error: %v", err)
		}
	}
	if w.conn != nil {
		if err := w.conn.Close(); err != nil {
			log.Printf("could not close connection. Error: %v", err)
		}
	}
}
