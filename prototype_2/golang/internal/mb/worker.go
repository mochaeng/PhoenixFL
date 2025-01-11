package mb

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/models"
	"github.com/mochaeng/phoenix-detector/internal/parser"
	"github.com/mochaeng/phoenix-detector/internal/stats"
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
	statsPath        string

	conn          *amqp.Connection
	channel       *amqp.Channel
	requestsQueue *amqp.Queue
	alertsQueue   *amqp.Queue
	requestsMsgs  <-chan amqp.Delivery
	confirmations <-chan amqp.Confirmation
}

func NewWorker(amqpURL, modelFile, statsPath string) *Worker {
	classifier, err := torchbidings.NewModel(modelFile)
	if err != nil {
		log.Panicf("could not load pytorch model. Error: %v\n", err)
	}
	return &Worker{
		name:        fmt.Sprintf("worker-%s", uuid.NewString()),
		stopConsume: make(chan bool),
		amqpURL:     amqpURL,
		classifier:  classifier,
		statsPath:   statsPath,
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

	confirmations, err := SetupPublisherConfirms(w.channel, 100)
	if err != nil {
		return err
	}
	w.confirmations = confirmations

	err = SetQoS(w.channel, 1)
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
	w.requestsQueue = requestsQueue

	alertsQueue, err := GetAlertsQueue(w.channel)
	if err != nil {
		return err
	}
	w.alertsQueue = alertsQueue

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
		return fmt.Errorf("could not consume from queue [requests_queue]. Error: %w\n", err)
	}
	w.requestsMsgs = requestsMsgs

	return nil
}

func (w *Worker) ConsumeRequestsQueue() {
	for {
		select {
		case <-w.stopConsume:
			return
		case delivery := <-w.requestsMsgs:
			var msg models.ClientRequest
			err := json.Unmarshal([]byte(delivery.Body), &msg)
			if err != nil {
				log.Printf("error parsing [ClientRequest] message. Error: %v\n", err)
				delivery.Nack(false, true)
				continue
			}

			convertedSentTimestamp := time.Unix(int64(msg.Timestamp), 0)
			transmissionAndQueueLatency := time.Now().Sub(convertedSentTimestamp)

			classificationStartTime := time.Now()
			isMalicious, err := w.classifier.PredictIsPositiveBinary(msg.Packet)
			if err != nil {
				log.Printf("prediction failed. Error: %v\n", err)
				delivery.Nack(false, true)
				continue
			}
			classificationLatency := time.Now().Sub(classificationStartTime)

			latency := transmissionAndQueueLatency + classificationLatency
			publishAlertStartTime := time.Now()
			classifiedPacket := models.ClassifiedPacket{
				Metadata:           msg.Metadata,
				ClassificationTime: classificationLatency,
				Latency:            latency,
				WorkerName:         w.name,
				IsMalicious:        isMalicious,
				Timestamp:          time.Now(),
			}
			classificationMessage, err := json.Marshal(classifiedPacket)
			if err != nil {
				log.Printf("failed to marshal classification message. Error: %v\n", err)
				delivery.Nack(false, true)
				continue
			}

			_, err = w.channel.PublishWithDeferredConfirm(
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
				continue
			}

			select {
			case confirmed := <-w.confirmations:
				if !confirmed.Ack {
					log.Print("message was not acknowledged by rabbitMQ\n")
					delivery.Nack(false, true)
					continue
				}
			case <-time.After(5 * time.Second):
				log.Print("timeout waiting for message confirmation\n")
				delivery.Nack(false, true)
				continue
			}

			publishAlertLatency := time.Now().Sub(publishAlertStartTime)
			inducedLatency := latency + publishAlertLatency
			w.latencies = append(w.latencies, float64(inducedLatency.Seconds()))
			w.processedPackets++

			delivery.Ack(false)
		}
	}
}

func (w *Worker) SaveLatencyStats() error {
	fileName := fmt.Sprintf("worker_%s_latencies.csv", w.name)
	filePath := filepath.Join(w.statsPath, fileName)

	writer, file, err := parser.CreateCSVWriter(filePath)
	if err != nil {
		return fmt.Errorf("could not create CSV writer. Error: %v\n", err)
	}
	defer file.Close()

	header := []string{"latency"}
	err = parser.WriteCSVRecord(writer, header)
	if err != nil {
		return fmt.Errorf("could not write headers. Erorr: %v\n", err)
	}

	for _, latency := range w.latencies {
		convertedLatency := strconv.FormatFloat(latency, 'f', -1, 64)
		err = parser.WriteCSVRecord(writer, []string{convertedLatency})
		if err != nil {
			return fmt.Errorf("could not write latency. Error: %v\n", err)
		}
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		return fmt.Errorf("could not flush writer. Error: %v\n", err)
	}

	return nil
}

func (w *Worker) SaveMetrics() error {
	fileName := fmt.Sprintf("worker_%s_metrics.json", w.name)
	filePath := filepath.Join(w.statsPath, fileName)

	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("could not create file. Error: %v\n", err)
	}
	defer file.Close()

	allMetrics := struct {
		Metrics          *stats.Metrics
		ProcessedPackets int
	}{
		Metrics:          stats.ComputeLatenciesMetrics(w.latencies),
		ProcessedPackets: w.processedPackets,
	}

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "	")
	err = encoder.Encode(allMetrics)
	if err != nil {
		return fmt.Errorf("could not write metrics to json. Error: %v\n", err)
	}

	return nil
}

func (w *Worker) Stop() {
	log.Printf("Stopping worker %s...\n", w.name)

	w.stopConsume <- true

	if w.channel != nil {
		if err := w.channel.Close(); err != nil {
			log.Printf("could not close channel. Error: %v\n", err)
		}
	}

	if w.conn != nil {
		if err := w.conn.Close(); err != nil {
			log.Printf("could not close connection. Error: %v\n", err)
		}
	}

	w.classifier.Delete()

	if err := w.SaveLatencyStats(); err != nil {
		log.Printf("could not write latencies. Error: %v\n", err)
	}

	if err := w.SaveMetrics(); err != nil {
		log.Printf("could not write metrics. Error: %v\n", err)
	}
}
