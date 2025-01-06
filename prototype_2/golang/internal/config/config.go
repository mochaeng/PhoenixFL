package config

const (
	AmqpURL = "amqp://guest:guest@localhost:5672/"

	PrefetchCount = 1

	PacketExchangeName = "packet"
	PacketExchangeType = "direct"

	RequestsQueueName       = "requests_queue"
	RequestsQueueRoutingKey = RequestsQueueName

	AlertsQueueName       = "alerts_queue"
	AlertsQueueRoutingKey = AlertsQueueName
)
