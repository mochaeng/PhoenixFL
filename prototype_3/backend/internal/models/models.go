package models

type MetadataResponse struct {
	SourceIpAddr  string `json:"ipv4_src_addr"`
	SourcePortNum int    `json:"l4_src_port"`
	DestIpAddr    string `json:"ipv4_dst_addr"`
	DestPortNum   int    `json:"l4_dst_port"`
}

type StatsResponse struct {
	TotalPackets   int64 `json:"total_packets"`
	TotalMalicious int64 `json:"total_malicious"`
}

type ClassifiedPacketResponse struct {
	ID                 string           `json:"id"`
	Metadata           MetadataResponse `json:"metadata"`
	Stats              StatsResponse    `json:"stats"`
	ClassificationTime float64          `json:"classification_time"`
	TotalTime          float64          `json:"total_time"`
	WorkerName         string           `json:"worker_name"`
	IsMalicious        bool             `json:"is_malicious"`
}
