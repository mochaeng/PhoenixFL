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

type SimulationThroughputResult struct {
	Timers     []float64 `json:"timers"`
	Mean       float64   `json:"mean"`
	Median     float64   `json:"median"`
	Quantile75 float64   `json:"quantile_75"`
	Quantile95 float64   `json:"quantile_95"`
	Quantile99 float64   `json:"quantile_99"`
}

type WorkerThroughputRecords struct {
	TotalFiles            int       `json:"total_files"`
	TotalProcessedPackets uint64    `json:"processed_packets"`
	Timers                []float64 `json:"timers"`
	Mean                  float64   `json:"mean"`
}
