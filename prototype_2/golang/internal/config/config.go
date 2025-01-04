package config

const (
	PrefetchCount = 1

	PacketExchangeName = "packet"
	PacketExchangeType = "direct"

	RequestsQueueName       = "requests_queue"
	RequestsQueueRoutingKey = RequestsQueueName

	AlertsQueueName       = "alerts_queue"
	AlertsQueueRoutingKey = AlertsQueueName
)
