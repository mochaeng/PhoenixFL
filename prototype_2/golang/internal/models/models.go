package models

type ClientRequest struct {
	Timestamp int64                  `json:"send_timestamp"`
	Metadata  map[string]interface{} `json:"metadata"`
	Packet    map[string]interface{} `json:"packet"`
}
