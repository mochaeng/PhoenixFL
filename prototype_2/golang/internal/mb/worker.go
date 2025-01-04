package mb

import (
	"context"
	"fmt"
	"log"

	"github.com/google/uuid"
	"github.com/mochaeng/phoenix-detector/internal/config"
	amqp "github.com/rabbitmq/amqp091-go"
)

type Worker struct {
	name          string
	amqpURL       string
	conn          *amqp.Connection
	channel       *amqp.Channel
	requestsQueue string
	alertsQueue   string
	latencies     []float64
	processedPkgs int
	ctx           context.Context
	cancel        context.CancelFunc
}

func NewWorker(amqpURL, requestsQueue, alertsQueue string) *Worker {
	ctx, cancel := context.WithCancel(context.Background())
	return &Worker{
		name:          fmt.Sprintf("worker_%s", uuid.NewString()),
		amqpURL:       amqpURL,
		requestsQueue: requestsQueue,
		alertsQueue:   alertsQueue,
		ctx:           ctx,
		cancel:        cancel,
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

	_, err = GetRequestsQueue(w.channel)
	if err != nil {
		return err
	}

	log.Printf("Declaring [alerts_queue] queue %s\n", config.RequestsQueueName)
	_, err = GetAlertsQueue(w.channel)
	if err != nil {
		return fmt.Errorf("could not declare [alerts_queue]. Error: %v\n", err)
	}

	return nil
}
