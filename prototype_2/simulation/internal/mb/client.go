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

const (
	publishInterval = 5 * time.Millisecond
)

type Client struct {
	amqpURL              string
	messages             []*models.ClientRequest
	currentPacket        uint64
	acked                uint64
	nacked               uint64
	hasFinished          atomic.Bool
	outstandingMsgsLimit int
	outstandingConfirms  map[uint64]*models.ClientRequest
	nackMsgs             *ConcurrentQueue[*models.ClientRequest]
	hasMessageLimit      bool
	messagesCountLimit   uint64

	conn    *amqp.Connection
	channel *amqp.Channel

	sync.RWMutex
}

// Use [messageLimitCount] == 0 if you don't want any limit,
// otherwise pass a value greather than 0
func NewClient(amqpURL string, messages []*models.ClientRequest, messageLimitCount uint64) *Client {
	outStandingLimit := 100
	return &Client{
		amqpURL:              amqpURL,
		messages:             messages,
		outstandingMsgsLimit: outStandingLimit,
		outstandingConfirms:  make(map[uint64]*models.ClientRequest),
		nackMsgs:             NewConcurrentQueue[*models.ClientRequest](),
		hasMessageLimit:      messageLimitCount != 0,
		messagesCountLimit:   messageLimitCount,
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

	confirmations, err := SetupPublisherConfirms(c.channel, c.outstandingMsgsLimit)
	if err != nil {
		return err
	}

	go c.handleAcknowledgments(confirmations)
	go c.handleNotAcknowledgedMsgs()

	return nil
}

func (c *Client) StartPublishing() {
	log.Println("Starting publishing messages")

	ticker := time.NewTicker(publishInterval)
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
			if err := c.publishRequestPacket(); err != nil {
				log.Printf("could not publish message. Error: %v\n", err)
			}
		}
	}
}

func (c *Client) publishRequestPacket() error {
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
	c.outstandingConfirms[sequenceNumber] = originalMessage

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
		return fmt.Errorf("failed to publish message: %w\n", err)
	}

	return nil
}

func (c *Client) handleNotAcknowledgedMsgs() {
	for {
		c.RLock()
		if c.hasFinished.Load() {
			c.RUnlock()
			return
		}
		c.RUnlock()

		c.nackMsgs.WaitForItem()
		for {
			msg, ok := c.nackMsgs.Dequeue()
			if !ok {
				break
			}
			c.retryMessage(msg)
		}
	}
}

func (c *Client) handleAcknowledgments(confirmations <-chan amqp.Confirmation) {
	for confirmation := range confirmations {
		if confirmation.Ack {
			// log.Printf("Message with delivery tag %d acknowledged\n", confirmation.DeliveryTag)
			c.Lock()
			c.acked++
			delete(c.outstandingConfirms, confirmation.DeliveryTag)
			if c.hasMessageLimit && c.acked >= c.messagesCountLimit {
				c.hasFinished.Store(true)
			}
			c.Unlock()
		} else {
			log.Printf(
				"Message with delivery tag [%d] not acknowledged. Retrying...\n",
				confirmation.DeliveryTag,
			)
			c.Lock()
			item, exists := c.outstandingConfirms[confirmation.DeliveryTag]
			if !exists || item == nil {
				log.Printf("no corresponding outstanding message for delivery tag [%d]", confirmation.DeliveryTag)
				c.Unlock()
				continue
			}
			c.nacked++
			c.nackMsgs.Enqueue(item)
			c.Unlock()
		}
	}
}

func (c *Client) retryMessage(originalMsg *models.ClientRequest) error {
	c.Lock()
	defer c.Unlock()

	if c.hasFinished.Load() {
		return nil
	}

	message := &models.ClientRequest{
		Timestamp: float64(time.Now().Unix()),
		Metadata:  originalMsg.Metadata,
		Packet:    originalMsg.Packet,
	}
	messageJSON, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message for retry: %w\n", err)
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
		return fmt.Errorf("failed to publish message during retry. Error: %w\n", err)
	}

	return nil
}

func (c *Client) Stop() {
	c.Lock()
	defer c.Unlock()

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
