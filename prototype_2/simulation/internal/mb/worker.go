package mb

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/mochaeng/phoenix-detector/internal/config"
	"github.com/mochaeng/phoenix-detector/internal/models"
	"github.com/mochaeng/phoenix-detector/internal/parser"
	"github.com/mochaeng/phoenix-detector/internal/stats"
	"github.com/mochaeng/phoenix-detector/internal/torchbidings"
	amqp "github.com/rabbitmq/amqp091-go"
)

type WorkerTimeMetrics struct {
	StartTime           time.Time
	EndTime             time.Time
	TotalProcessingTime time.Duration
}

type Worker struct {
	name             string
	amqpURL          string
	latencies        []float64
	processedPackets int
	stopConsume      chan bool
	classifier       *torchbidings.Classifier
	statsPath        string
	timeMetrics      *WorkerTimeMetrics
	isIdle           atomic.Bool
	idleTimeout      *time.Duration
	hasFinished      atomic.Bool
	acked            uint64

	conn          *amqp.Connection
	channel       *amqp.Channel
	requestsQueue *amqp.Queue
	alertsQueue   *amqp.Queue
	requestsMsgs  <-chan amqp.Delivery
	confirmations <-chan amqp.Confirmation
}

func NewWorker(amqpURL, modelFile, statsPath string, idleTimeout *time.Duration) *Worker {
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
		timeMetrics: &WorkerTimeMetrics{},
		idleTimeout: idleTimeout,
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

func (w *Worker) ConsumeRequestsQueue() {
	w.timeMetrics.StartTime = time.Now()

	var idleTimer *time.Timer
	if w.idleTimeout != nil {
		idleTimer = time.NewTimer(*w.idleTimeout)
	}

	for {
		select {
		case <-w.stopConsume:
			return
		case <-func() <-chan time.Time {
			if idleTimer != nil {
				return idleTimer.C
			}
			return nil
		}():
			if !w.isIdle.Load() {
				log.Printf("worker [%s] has been idle for [%v], processed [%d] packets\n", w.name, w.idleTimeout, w.processedPackets)
				w.putWorkerInIdleMode()
				return
			}
			idleTimer.Reset(*w.idleTimeout)
		case delivery, ok := <-w.requestsMsgs:
			if !ok {
				continue
			}
			if w.hasFinished.Load() {
				return
			}

			if idleTimer != nil {
				w.resetIdlerTimer(idleTimer)
			}

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
			alertMsg := models.ClassifiedPacket{
				Metadata:           msg.Metadata,
				ClassificationTime: classificationLatency,
				Latency:            latency,
				WorkerName:         w.name,
				IsMalicious:        isMalicious,
				Timestamp:          time.Now(),
			}
			if err := w.publishAlert(&alertMsg); err != nil {
				log.Println(err)
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

func (w *Worker) Stop() {
	if w.hasFinished.Load() {
		return
	}

	w.timeMetrics.EndTime = time.Now()
	w.timeMetrics.TotalProcessingTime = w.timeMetrics.EndTime.Sub(w.timeMetrics.StartTime)

	log.Printf("Stopping worker [%s]...\n", w.name)

	// if worker isIdle it cannot consuming from the channel, otherwise would block
	if !w.isIdle.Load() {
		w.stopConsume <- true
	}

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

	if err := w.saveLatencyStats(); err != nil {
		log.Printf("could not write latencies. Error: %v\n", err)
	}

	if err := w.saveMetrics(); err != nil {
		log.Printf("could not write metrics. Error: %v\n", err)
	}

	w.hasFinished.Store(true)
	log.Print("Worker has finished\n")
}

func (w *Worker) SetupWorker() error {
	if w.channel == nil {
		return ErrInvalidChannel
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

func (w *Worker) publishAlert(alertMsg *models.ClassifiedPacket) error {
	classificationMessage, err := json.Marshal(alertMsg)
	if err != nil {
		return fmt.Errorf("failed to marshal classification message. Error: %v\n", err)
	}

	sequenceNumber := w.channel.GetNextPublishSeqNo()

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
		return fmt.Errorf("failed to publish message. Error: %v\n", err)
	}

	WaitForPublishConfirmation(w.confirmations, sequenceNumber, 5*time.Second)

	return nil
}

func (w *Worker) saveLatencyStats() error {
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

func (w *Worker) saveMetrics() error {
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
		TotalTimer       time.Duration
	}{
		Metrics:          stats.ComputeLatenciesMetrics(w.latencies),
		ProcessedPackets: w.processedPackets,
		TotalTimer:       time.Duration(w.timeMetrics.EndTime.Sub(w.timeMetrics.StartTime).Seconds()),
	}

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "	")
	err = encoder.Encode(allMetrics)
	if err != nil {
		return fmt.Errorf("could not write metrics to json. Error: %v\n", err)
	}

	return nil
}

func (w *Worker) putWorkerInIdleMode() {
	w.isIdle.Store(true)
	// w.timeMetrics.EndTime = time.Now()
	// w.timeMetrics.TotalProcessingTime = w.timeMetrics.EndTime.Sub(w.timeMetrics.StartTime)
	w.Stop()
}

func (w *Worker) resetIdlerTimer(idleTimer *time.Timer) {
	if idleTimer == nil {
		return
	}

	if !idleTimer.Stop() {
		select {
		case <-idleTimer.C:
		default:
		}
	}
	idleTimer.Reset(*w.idleTimeout)
	w.isIdle.Store(false)
}
