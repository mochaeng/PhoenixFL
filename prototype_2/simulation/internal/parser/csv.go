package parser

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/mochaeng/phoenix-detector/internal/models"
)

func CreateCSVWriter(filePath string) (*csv.Writer, *os.File, error) {
	f, err := os.Create(filePath)
	if err != nil {
		return nil, nil, err
	}
	return csv.NewWriter(f), f, nil
}

func WriteCSVRecord(writer *csv.Writer, record []string) error {
	err := writer.Write(record)
	if err != nil {
		return fmt.Errorf("could not write record. Error: %v\n", err)
	}
	return nil
}

func ParsePacketsCSV(filePath string, columnsToRemove []string) ([]*models.ClientRequest, error) {
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

	var messages []*models.ClientRequest
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		metadata := make(map[string]string)
		packetValues := make([]float32, 0, len(record))

		for i, value := range record {
			column := headers[i]
			if _, isRemove := removeSet[column]; isRemove {
				metadata[column] = value
			} else {
				floatValue, err := strconv.ParseFloat(value, 32)
				if err != nil {
					log.Printf("could not convert string value to float. Error: %v\n", err)
				}
				packetValues = append(packetValues, float32(floatValue))
			}
		}

		sourcePort, err := strconv.ParseInt(metadata["L4_SRC_PORT"], 10, 64)
		if err != nil {
			log.Printf("could not convert string value to int. Error: %v\n", err)
		}
		destPort, err := strconv.ParseInt(metadata["L4_DST_PORT"], 10, 64)
		if err != nil {
			log.Printf("could not convert string value to int. Error: %v\n", err)
		}

		messages = append(messages, &models.ClientRequest{
			Metadata: models.MetadataRequest{
				SourceIP:   metadata["IPV4_SRC_ADDR"],
				SourcePort: int(sourcePort),
				DestIP:     metadata["IPV4_DST_ADDR"],
				DestPort:   int(destPort),
			},
			Packet: packetValues,
		})
	}

	return messages, nil
}
