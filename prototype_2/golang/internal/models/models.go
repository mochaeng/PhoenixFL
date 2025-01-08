package models

import "time"

type MetadataRequest struct {
	SourceIP   string `json:"IPV4_SRC_ADDR"`
	SourcePort int    `json:"L4_SRC_PORT"`
	DestIP     string `json:"IPV4_DST_ADDR"`
	DestPort   int    `json:"L4_DST_PORT"`
}

type ClientRequest struct {
	Timestamp time.Time `json:"send_timestamp"`
	// Metadata  map[string]interface{} `json:"metadata"`
	Metadata MetadataRequest `json:"metadata"`
	// Packet   map[string]float64 `json:"packet"`
	Packet []float32 `json:"packet"`
}

type ClassifiedPacket struct {
	Metadata           MetadataRequest `json:"metadata"`
	ClassificationTime time.Duration   `json:"classification_time"`
	Latency            time.Duration   `json:"latency"`
	WorkerName         string          `json:"worker_name"`
	IsMalicious        bool            `json:"is_malicious"`
	Timestamp          time.Time       `json:"timestamp"`
}
