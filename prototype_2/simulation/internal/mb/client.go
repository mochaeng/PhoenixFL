package mb

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/models"
	amqp "github.com/rabbitmq/amqp091-go"
)

type Client struct {
	amqpURL              string
	messages             []*models.ClientRequest
	currentPacket        uint64
	acked                uint64
	nacked               uint64
	hasFinished          atomic.Bool
	outstandingMsgsLimit int
	hasMessageLimit      bool
	messagesCountLimit   uint64
	publishInterval      time.Duration

	conn          *amqp.Connection
	channel       *amqp.Channel
	confirmations <-chan amqp.Confirmation

	sync.RWMutex
}

// Use [messageLimitCount] == 0 if you don't want any limit,
// otherwise pass a value greather than 0
func NewClient(amqpURL string, messages []*models.ClientRequest, messageLimitCount uint64, publishInterval time.Duration) *Client {
	outStandingLimit := 1
	return &Client{
		amqpURL:              amqpURL,
		messages:             messages,
		outstandingMsgsLimit: outStandingLimit,
		hasMessageLimit:      messageLimitCount != 0,
		messagesCountLimit:   messageLimitCount,
		publishInterval:      publishInterval,
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

func (c *Client) SetupClient() error {
	confirmations, err := SetupPublisherConfirms(c.channel, c.outstandingMsgsLimit)
	if err != nil {
		return err
	}
	c.confirmations = confirmations

	err = SetQoS(c.channel, 1)
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

	ticker := time.NewTicker(c.publishInterval)
	defer ticker.Stop()

	for {
		c.RLock()
		if c.hasFinished.Load() {
			c.RUnlock()
			return
		}
		c.RUnlock()

		select {
		case <-ticker.C:
			if err := c.PublishRequestPacket(); err != nil {
				log.Printf("could not publish message. Error: %v\n", err)
				c.Stop()
				return
			}
		}
	}
}

func (c *Client) PublishRequestPacket() error {
	c.Lock()
	defer c.Unlock()

	if c.hasFinished.Load() {
		return nil
	}

	originalMessage := c.messages[c.currentPacket%uint64(len(c.messages))]
	message := &models.ClientRequest{
		Timestamp: float64(time.Now().Unix()),
		Metadata:  originalMessage.Metadata,
		Packet:    originalMessage.Packet,
	}
	messageJSON, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w\n", err)
	}
	c.currentPacket++

	sequenceNumber := c.channel.GetNextPublishSeqNo()

	_, err = c.channel.PublishWithDeferredConfirm(
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
		return fmt.Errorf("failed to publish message: %w\n", err)
	}

	WaitForPublishConfirmation(c.confirmations, sequenceNumber, 5*time.Second)

	c.acked++
	if c.hasMessageLimit && c.acked >= c.messagesCountLimit {
		c.hasFinished.Store(true)
	}

	return nil
}

func (c *Client) Stop() {
	c.Lock()
	defer c.Unlock()

	if c.hasFinished.Load() {
		return
	}

	c.hasFinished.Store(true)

	log.Printf("number of acked packets: %d\n", c.acked)

	if c.channel != nil {
		log.Println("closing channel from client")
		_ = c.channel.Close()
	}

	if c.conn != nil {
		log.Println("closing connection from client")
		_ = c.conn.Close()
	}
}
