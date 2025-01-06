package models

type MetadataRequest struct {
	SourceIP   string `json:"IPV4_SRC_ADDR"`
	SourcePort int    `json:"L4_SRC_PORT"`
	DestIP     string `json:"IPV4_DST_ADDR"`
	DestPort   int    `json:"L4_DST_PORT"`
}

type ClientRequest struct {
	Timestamp int64 `json:"send_timestamp"`
	// Metadata  map[string]interface{} `json:"metadata"`
	Metadata MetadataRequest    `json:"metadata"`
	Packet   map[string]float64 `json:"packet"`
}
