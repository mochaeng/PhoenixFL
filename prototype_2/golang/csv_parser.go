package main

import (
	"encoding/csv"
	"fmt"
	"os"
)

func ParseCSV(filePath string, columnsToRemove []string) ([]*ClientRequest, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v\n", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	headers, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read headers: %v\n", err)
	}

	removeSet := make(map[string]struct{}, len(columnsToRemove))
	for _, col := range columnsToRemove {
		removeSet[col] = struct{}{}
	}

	var messages []*ClientRequest
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		metadata := make(map[string]interface{})
		packet := make(map[string]interface{})

		for i, value := range record {
			column := headers[i]
			if _, isRemove := removeSet[column]; isRemove {
				metadata[column] = value
			} else {
				packet[column] = value
			}
		}

		messages = append(messages, &ClientRequest{
			Metadata: metadata,
			Packet:   packet,
		})
	}

	return messages, nil
}
