package config

const (
	AmqpURL   = "amqp://guest:guest@localhost:5672/"
	ModelPath = "../../data/fedmedian_model.pt"

	PrefetchCount = 1

	PacketExchangeName = "packet"
	PacketExchangeType = "direct"

	RequestsQueueName       = "requests_queue"
	RequestsQueueRoutingKey = RequestsQueueName

	AlertsQueueName       = "alerts_queue"
	AlertsQueueRoutingKey = AlertsQueueName
)
