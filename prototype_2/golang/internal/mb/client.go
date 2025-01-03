package mb

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/mochaeng/phoenix-detector/internal/models"
	amqp "github.com/rabbitmq/amqp091-go"
)

type Client struct {
	amqpURL       string
	messages      []*models.ClientRequest
	connection    *amqp.Connection
	channel       *amqp.Channel
	currentPacket int64
	acked         int64
	nacked        int64
	messageNumber int64
	stopping      bool
	stopChan      chan struct{}
	ctx           context.Context
	sync.RWMutex
}

const (
	exchangeName    = "packet"
	exchangeType    = "direct"
	publishInterval = 1 * time.Millisecond
	queueName       = "requests_queue"
	routingKey      = queueName
)

func NewClient(url string, messages []*models.ClientRequest, ctx context.Context) *Client {
	return &Client{
		amqpURL:  url,
		messages: messages,
		stopChan: make(chan struct{}),
		ctx:      ctx,
	}
}

func (c *Client) Connect() error {
	fmt.Printf("Connecting to %s\n", c.amqpURL)
	conn, err := amqp.Dial(c.amqpURL)
	if err != nil {
		return err
	}
	c.connection = conn
	return nil
}

func (c *Client) SetupChannel() error {
	fmt.Println("Creating a new channel")
	ch, err := c.connection.Channel()
	if err != nil {
		return err
	}
	c.channel = ch

	prefetchCount := 2
	log.Printf("Setting QoS with prefetch count %d\n", prefetchCount)
	err = c.channel.Qos(
		prefetchCount,
		0,
		false,
	)
	if err != nil {
		return fmt.Errorf("failed to set QoS: %w", err)
	}

	log.Printf("Declaring exchange %s\n", exchangeName)
	err = c.channel.ExchangeDeclare(
		exchangeName,
		exchangeType,
		true,  // durable
		false, // auto-deleted
		false, // internal
		false, // no-wait
		amqp.Table{},
	)
	if err != nil {
		return err
	}

	log.Printf("Declaring queue %s\n", queueName)
	_, err = c.channel.QueueDeclare(
		queueName,
		true,  // durable
		false, // auto-deleted
		false, // exclusive
		false, // no-wait
		amqp.Table{},
	)
	if err != nil {
		return err
	}

	log.Printf("Binding queue %s to exchange %s with routing key %s\n", queueName, exchangeName, routingKey)
	err = c.channel.QueueBind(
		queueName,
		routingKey,
		exchangeName,
		false, // no-wait
		amqp.Table{},
	)
	return err
}

func (c *Client) StartPublishing() {
	fmt.Println("Starting publishing messages")

	ticker := time.NewTicker(publishInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			fmt.Println("Stopping publishing")
			return
		case <-ticker.C:
			go c.publishMessage()
		}
	}
}

func (c *Client) publishMessage() {
	c.Lock()
	defer c.Unlock()

	if c.stopping {
		return
	}

	originalMessage := c.messages[c.currentPacket%int64(len(c.messages))]
	message := &models.ClientRequest{
		Timestamp: time.Now().Unix(),
		Metadata:  originalMessage.Metadata,
		Packet:    originalMessage.Packet,
	}

	messageJSON, err := json.Marshal(message)
	if err != nil {
		fmt.Printf("Failed to marshal message: %v\n", err)
		return
	}

	err = c.channel.Publish(
		exchangeName,
		routingKey,
		false, // mandatory
		false, // immediate
		amqp.Publishing{
			ContentType: "application/json",
			Body:        messageJSON,
		},
	)
	if err != nil {
		fmt.Printf("Failed to publish message: %v\n", err)
		c.nacked++
		return
	}

	c.currentPacket++
	c.messageNumber++
	c.acked++
	// fmt.Printf("Published message # %d\n", c.messageNumber)
}

func (c *Client) SetupPublisherConfirms() error {
	log.Println("Enabling publisher confirmations")
	if err := c.channel.Confirm(false); err != nil {
		return fmt.Errorf("failed to enable publisher confirmations: %v", err)
	}

	confirmations := c.channel.NotifyPublish(make(chan amqp.Confirmation, 1))
	go c.handleAcknowledgments(confirmations)
	return nil
}

func (c *Client) handleAcknowledgments(confirmations <-chan amqp.Confirmation) {
	for confirmation := range confirmations {
		select {
		case <-c.ctx.Done():
			fmt.Println("Stopping handling acks")
			return
		default:
			if confirmation.Ack {
				// log.Printf("Message with delivery tag %d acknowledged\n", confirmation.DeliveryTag)
				c.Lock()
				c.acked++
				c.Unlock()
			} else {
				log.Printf("Message with delivery tag %d not acknowledged. Retrying...\n", confirmation.DeliveryTag)
				c.retryMessage(confirmation.DeliveryTag)
			}
		}
	}
}

func (c *Client) retryMessage(deliveryTag uint64) {
	c.Lock()
	defer c.Unlock()

	message := c.messages[(deliveryTag-1)%uint64(len(c.messages))]
	messageJSON, err := json.Marshal(message)
	if err != nil {
		log.Printf("Failed to marshal message for retry: %v\n", err)
		c.nacked++
		return
	}

	err = c.channel.Publish(
		exchangeName,
		routingKey,
		false, // mandatory
		false, // immediate
		amqp.Publishing{
			ContentType: "application/json",
			Body:        messageJSON,
		},
	)
	if err != nil {
		log.Printf("Failed to publish message during retry: %v\n", err)
		c.nacked++
		return
	}
	log.Printf("Retried message with delivery tag %d successfully\n", deliveryTag)
}

func (c *Client) Stop() {
	c.Lock()
	defer c.Unlock()

	fmt.Println("was this even called??")
	fmt.Println("Stopping client")
	c.stopping = true
	close(c.stopChan)

	if c.channel != nil {
		fmt.Println("Closing channel")
		_ = c.channel.Close()
	}

	if c.connection != nil {
		fmt.Println("Closing connection")
		_ = c.connection.Close()
	}
}
