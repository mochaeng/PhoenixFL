package models

type MetadataResponse struct {
	SourceIpAddr  string `json:"IPV4_SRC_ADDR"`
	SourcePortNum int    `json:"L4_SRC_PORT"`
	DestIpAddr    string `json:"IPV4_DST_ADDR"`
	DestPortNum   int    `json:"L4_DST_PORT"`
}

type ClassifiedPacketResponse struct {
	Metadata           MetadataResponse `json:"metadata"`
	ClassificationTime float64          `json:"classification_time"`
	TotalTime          float64          `json:"total_time"`
	WorkerName         string           `json:"worker_name"`
	IsMalicious        bool             `json:"is_malicious"`
}
