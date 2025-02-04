package models

type IpCount struct {
	Address string `json:"address"`
	Count   int64  `json:"count"`
}

type Packet struct {
	Metadata           MetadataResponse `json:"metadata"`
	ClassificationTime float64          `json:"classification_time"`
	Latency            float64          `json:"latency"`
	WorkerName         string           `json:"worker_name"`
	IsMalicious        bool             `json:"is_malicious"`
	Timestamp          int64            `json:"timestamp"`
}

type MetadataResponse struct {
	SourceIpAddr  string `json:"ipv4_src_addr"`
	SourcePortNum int    `json:"l4_src_port"`
	DestIpAddr    string `json:"ipv4_dst_addr"`
	DestPortNum   int    `json:"l4_dst_port"`
}

type StatsResponse struct {
	TotalPackets          int64             `json:"total_packets"`
	TotalMalicious        int64             `json:"total_malicious"`
	AvgLatency            float64           `json:"avg_latency"`
	AvgClassificationTime float64           `json:"avg_classification_time"`
	MaliciousIps          []*ItemCount[int] `json:"malicious_ips"`
	TargetedIps           []*ItemCount[int] `json:"targeted_ips"`
}

type PacketWithStatsResponse struct {
	ID      string         `json:"id"`
	Stats   *StatsResponse `json:"stats"`
	*Packet `json:"packet_info"`
}
