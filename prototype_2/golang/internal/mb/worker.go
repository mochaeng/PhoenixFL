package mb

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/models"
	"github.com/mochaeng/phoenix-detector/internal/torchbidings"
	amqp "github.com/rabbitmq/amqp091-go"
)

type Worker struct {
	name             string
	amqpURL          string
	latencies        []float64
	processedPackets int
	stopConsume      chan bool
	classifier       *torchbidings.Classifier

	conn          *amqp.Connection
	channel       *amqp.Channel
	requestsQueue *amqp.Queue
	alertsQueue   *amqp.Queue
	requestsMsgs  <-chan amqp.Delivery
}

func NewWorker(amqpURL string, modelFile string) *Worker {
	classifier, err := torchbidings.NewModel(modelFile)
	if err != nil {
		log.Panicf("could not load pytorch model. Error: %v\n", err)
	}
	return &Worker{
		name:        fmt.Sprintf("worker_%s", uuid.NewString()),
		stopConsume: make(chan bool),
		amqpURL:     amqpURL,
		classifier:  classifier,
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

func (w *Worker) SetupWorker() error {
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

	alertsQueue, err := GetAlertsQueue(w.channel)
	if err != nil {
		return err
	}

	err = BindRequestsQueueWithPacketExchange(w.channel)
	if err != nil {
		return err
	}

	err = BindAlertsQueueWithPacketExchange(w.channel)
	if err != nil {
		return err
	}

	requestsMsgs, err := w.channel.Consume(
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

	w.requestsQueue = requestsQueue
	w.alertsQueue = alertsQueue
	w.requestsMsgs = requestsMsgs

	return nil
}

func (w *Worker) ConsumeRequestsRequeue() {
	for {
		select {
		case <-w.stopConsume:
			return
		case delivery := <-w.requestsMsgs:
			processingStartTime := time.Now()
			var msg models.ClientRequest
			err := json.Unmarshal([]byte(delivery.Body), &msg)
			if err != nil {
				log.Printf("error parsing [ClientRequest] message. Error: %v\n", err)
				delivery.Nack(false, true)
				continue
			}
			// log.Printf("%v+\n", msg.Timestamp)

			transmissionAndQueueLatency := processingStartTime.Sub(msg.Timestamp)
			classificationStartTime := time.Now()

			// pytorch classification...
			classificationLatency := classificationStartTime.Sub(time.Now())
			totalLatency := transmissionAndQueueLatency + classificationLatency

			classifiedPacket := models.ClassifiedPacket{
				Metadata:           msg.Metadata,
				ClassificationTime: classificationLatency,
				Latency:            totalLatency,
				WorkerName:         w.name,
				IsMalicious:        false,
				Timestamp:          time.Now(),
			}
			classificationMessage, err := json.Marshal(classifiedPacket)
			if err != nil {
				log.Printf("failed to marshal classification message. Error: %v\n", err)
				delivery.Nack(false, true)
				continue
			}

			err = w.channel.Publish(
				config.PacketExchangeName,
				config.AlertsQueueRoutingKey,
				false, // mandatory
				false, // immediate
				amqp.Publishing{
					ContentType: "application/json",
					Body:        classificationMessage,
				},
			)
			if err != nil {
				log.Printf("failed to publish message: %v\n", err)
				delivery.Nack(false, true)
			}

			w.latencies = append(w.latencies, float64(totalLatency.Milliseconds()))
			w.processedPackets++
			delivery.Ack(false)
		}
	}
}

func (w *Worker) Stop() {
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
