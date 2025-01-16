package mb

import (
	"fmt"
	"log"
	"time"

	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/models"
	amqp "github.com/rabbitmq/amqp091-go"
)

func SetQoS(channel *amqp.Channel, prefetchCount int) error {
	if channel == nil {
		return ErrInvalidChannel
	}
	log.Printf("Setting QoS with prefetch count %d\n", prefetchCount)
	err := channel.Qos(prefetchCount, 0, false)
	if err != nil {
		return fmt.Errorf("could not set QoS. Error: %v\n", err)
	}
	return nil
}

func PacketExchangeDeclare(channel *amqp.Channel) error {
	if channel == nil {
		return ErrInvalidChannel
	}
	log.Println("Declaring exchange [packet]")
	err := channel.ExchangeDeclare(
		config.PacketExchangeName,
		config.PacketExchangeType,
		true,  // durable
		false, // auto-deleted
		false, // internal
		false, // no-wait
		amqp.Table{},
	)
	if err != nil {
		return fmt.Errorf("could not declare exchange [packet]. Error: %v\n", err)
	}
	return nil
}

func GetRequestsQueue(channel *amqp.Channel) (*amqp.Queue, error) {
	if channel == nil {
		return nil, ErrInvalidChannel
	}
	log.Println("Declaring queue [requests_queue]")
	queue, err := channel.QueueDeclare(
		config.RequestsQueueName,
		true,  // durable
		false, // auto-deleted
		false, // exclusive
		false, // no-wait
		amqp.Table{},
	)
	if err != nil {
		return nil, fmt.Errorf("could not declare [requests_queue]. Error: %v\n", err)
	}
	return &queue, nil
}

func GetAlertsQueue(channel *amqp.Channel) (*amqp.Queue, error) {
	if channel == nil {
		return nil, ErrInvalidChannel
	}
	log.Println("Declaring [alerts_queue] queue")
	queue, err := channel.QueueDeclare(
		config.AlertsQueueName,
		true,  // durable
		false, // auto-deleted
		false, // exclusive
		false, // no-wait
		amqp.Table{},
	)
	if err != nil {
		return nil, fmt.Errorf("could not declare [alerts_queue]. Error: %v\n", err)
	}
	return &queue, nil
}

func bindQueueWithExchange(channel *amqp.Channel, queueName, routingKey, exchangeName string) error {
	if channel == nil {
		return ErrInvalidChannel
	}
	log.Printf("Binding queue %s to exchange %s with routing key %s\n", queueName, exchangeName, routingKey)
	err := channel.QueueBind(
		queueName,
		routingKey,
		exchangeName,
		false, // no-wait
		amqp.Table{},
	)
	if err != nil {
		return fmt.Errorf("could not bind queue [%s] with exchange [%s]. Error: %v\n", queueName, exchangeName, err)
	}
	return nil
}

func BindRequestsQueueWithPacketExchange(channel *amqp.Channel) error {
	err := bindQueueWithExchange(
		channel,
		config.RequestsQueueName,
		config.RequestsQueueRoutingKey,
		config.PacketExchangeName,
	)
	if err != nil {
		return err
	}
	return nil
}

func BindAlertsQueueWithPacketExchange(channel *amqp.Channel) error {
	err := bindQueueWithExchange(
		channel,
		config.AlertsQueueName,
		config.AlertsQueueRoutingKey,
		config.PacketExchangeName,
	)
	if err != nil {
		return err
	}
	return nil
}

func SetupPublisherConfirms(channel *amqp.Channel, chanSize int) (<-chan amqp.Confirmation, error) {
	log.Println("Enabling publisher confirmations")
	if err := channel.Confirm(false); err != nil {
		return nil, fmt.Errorf("failed to enable publisher confirmations. Error: %w\n", err)
	}
	confirmations := channel.NotifyPublish(make(chan amqp.Confirmation, chanSize))
	return confirmations, nil
}

func WaitForPublishConfirmation(confirmations <-chan amqp.Confirmation, sequenceNumber uint64, timeout time.Duration) error {
	select {
	case confirmed := <-confirmations:
		if !confirmed.Ack {
			return ErrNackedMessage
		}
		if confirmed.DeliveryTag != sequenceNumber {
			return fmt.Errorf(
				"expected delivery tag [%d], have [%d]\n",
				sequenceNumber,
				confirmed.DeliveryTag,
			)
		}
		return nil
	case <-time.After(timeout):
		return ErrPublishConfirmTimeout
	}
}

func GetReadyClient(url string, messages []*models.ClientRequest, messageLimit uint64, publishInterval time.Duration) (*Client, error) {
	client := NewClient(url, messages, 0, publishInterval)
	err := client.Connect()
	if err != nil {
		return nil, fmt.Errorf("failed to connect: %v\n", err)
	}
	err = client.SetupClient()
	if err != nil {
		log.Panicf("Failed to set up client: %v\n", err)
	}
	return client, nil
}

func GetReadyWorkers(url, modelPath string, numWorkers int, idleTimeout *time.Duration) ([]*Worker, error) {
	workers := make([]*Worker, 0, numWorkers)
	for i := 0; i < numWorkers; i++ {
		worker := NewWorker(url, modelPath, "../../../data/workers", idleTimeout)
		if err := worker.Connect(); err != nil {
			return nil, fmt.Errorf("could not connect worker to rabbitMQ. Error: %w\n", err)
		}
		if err := worker.SetupWorker(); err != nil {
			return nil, fmt.Errorf("could not setup worker. Error: %w\n", err)
		}
		// go worker.ConsumeRequestsQueue()
		workers = append(workers, worker)
	}
	return workers, nil
}
