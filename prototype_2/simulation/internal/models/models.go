package models

import "time"

type MetadataRequest struct {
	SourceIP   string `json:"IPV4_SRC_ADDR"`
	SourcePort int    `json:"L4_SRC_PORT"`
	DestIP     string `json:"IPV4_DST_ADDR"`
	DestPort   int    `json:"L4_DST_PORT"`
}

type ClientRequest struct {
	Timestamp float64         `json:"send_timestamp"`
	Metadata  MetadataRequest `json:"metadata"`
	Packet    []float32       `json:"packet"`
}

type ClassifiedPacket struct {
	Metadata           MetadataRequest `json:"metadata"`
	ClassificationTime time.Duration   `json:"classification_time"`
	Latency            time.Duration   `json:"latency"`
	WorkerName         string          `json:"worker_name"`
	IsMalicious        bool            `json:"is_malicious"`
	Timestamp          time.Time       `json:"timestamp"`
}

type RecordThroughput struct {
	ProcessedPackets uint64  `json:"processed_packets"`
	Timer            float64 `json:"total_timer"`
}

type AggregatedThroughput struct {
	TotalFiles            int
	TotalProcessedPackets uint64
	Timers                []float64
	Mean                  float64
	Median                float64
	Quantile75            float64
	Quantile95            float64
	Quantile99            float64
}
