package mb

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/models"
	amqp "github.com/rabbitmq/amqp091-go"
)

const (
	publishInterval = 5 * time.Millisecond
)

type Client struct {
	amqpURL       string
	messages      []*models.ClientRequest
	currentPacket int64
	acked         int64
	nacked        int64
	messageNumber int64
	hasStopped    bool

	conn    *amqp.Connection
	channel *amqp.Channel

	sync.RWMutex
}

func NewClient(url string, messages []*models.ClientRequest) *Client {
	return &Client{
		amqpURL:  url,
		messages: messages,
	}
}

func (c *Client) Connect() error {
	log.Printf("Connecting to %s\n", c.amqpURL)
	conn, err := amqp.Dial(c.amqpURL)
	if err != nil {
		return err
	}
	c.conn = conn

	channel, err := c.conn.Channel()
	if err != nil {
		return err
	}
	c.channel = channel

	return nil
}

func (c *Client) SetupRabbitMQ() error {
	err := SetQoS(c.channel, 1)
	if err != nil {
		return err
	}

	err = PacketExchangeDeclare(c.channel)
	if err != nil {
		return err
	}

	_, err = GetRequestsQueue(c.channel)
	if err != nil {
		return err
	}

	err = BindRequestsQueueWithPacketExchange(c.channel)
	if err != nil {
		return err
	}

	return nil
}

func (c *Client) StartPublishing() {
	log.Println("Starting publishing messages")

	ticker := time.NewTicker(publishInterval)
	defer ticker.Stop()

	for {
		if c.hasStopped {
			return
		}
		select {
		case <-ticker.C:
			c.publishMessage()
		}
	}
}

func (c *Client) publishMessage() {
	c.Lock()
	defer c.Unlock()

	if c.hasStopped {
		return
	}

	originalMessage := c.messages[c.currentPacket%int64(len(c.messages))]
	message := &models.ClientRequest{
		Timestamp: time.Now(),
		Metadata:  originalMessage.Metadata,
		Packet:    originalMessage.Packet,
	}

	messageJSON, err := json.Marshal(message)
	if err != nil {
		log.Printf("Failed to marshal message: %v\n", err)
		return
	}

	err = c.channel.Publish(
		config.PacketExchangeName,
		config.RequestsQueueRoutingKey,
		false, // mandatory
		false, // immediate
		amqp.Publishing{
			ContentType: "application/json",
			Body:        messageJSON,
		},
	)
	if err != nil {
		log.Printf("Failed to publish message: %v\n", err)
		c.nacked++
		return
	}

	c.currentPacket++
	c.messageNumber++
	// log.Printf("Published message # %d\n", c.messageNumber)
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

func (c *Client) retryMessage(deliveryTag uint64) {
	c.Lock()
	defer c.Unlock()

	if c.hasStopped {
		return
	}

	originalMessage := c.messages[(deliveryTag-1)%uint64(len(c.messages))]
	message := &models.ClientRequest{
		Timestamp: time.Now(),
		Metadata:  originalMessage.Metadata,
		Packet:    originalMessage.Packet,
	}
	messageJSON, err := json.Marshal(message)
	if err != nil {
		log.Printf("Failed to marshal message for retry: %v\n", err)
		c.nacked++
		return
	}

	err = c.channel.Publish(
		config.PacketExchangeName,
		config.RequestsQueueRoutingKey,
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

	c.hasStopped = true

	log.Printf("Number of ackked packets: %d\n", c.acked)

	if c.channel != nil {
		log.Println("Closing channel from client")
		_ = c.channel.Close()
	}

	if c.conn != nil {
		log.Println("Closing connection from client")
		_ = c.conn.Close()
	}
}
