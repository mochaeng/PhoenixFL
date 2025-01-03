package mb

import (
	"context"

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
	// ctx, cancel := context.WithCancel(context.Background())
	// return &Worker{
	// 	name: fmt.Sprintf("worker_", uuid.NewString()),
	// }
	return nil
}
