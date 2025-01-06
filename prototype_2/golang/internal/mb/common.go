package mb

import (
	"fmt"
	"log"

	"github.com/mochaeng/phoenix-detector/internal/config"
	amqp "github.com/rabbitmq/amqp091-go"
)

func SetQoS(channel *amqp.Channel, prefetchCount int) error {
	if channel == nil {
		return config.ErrInvalidChannel
	}
	log.Printf("Setting QoS with prefetch count %d\n", config.PrefetchCount)
	err := channel.Qos(config.PrefetchCount, 0, false)
	if err != nil {
		return fmt.Errorf("could not set QoS. Error: %v\n", err)
	}
	return nil
}

func PacketExchangeDeclare(channel *amqp.Channel) error {
	if channel == nil {
		return config.ErrInvalidChannel
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
		return nil, config.ErrInvalidChannel
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
		return nil, config.ErrInvalidChannel
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
		return config.ErrInvalidChannel
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
