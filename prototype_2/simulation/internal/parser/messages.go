package parser

import (
	"fmt"

	"github.com/mochaeng/phoenix-detector/internal/models"
)

func GetMessages(csvPath string) ([]*models.ClientRequest, error) {
	columnsToRemove := []string{
		"IPV4_SRC_ADDR",
		"IPV4_DST_ADDR",
		"L4_SRC_PORT",
		"L4_DST_PORT",
	}
	messages, err := ParsePacketsCSV(csvPath, columnsToRemove)
	if err != nil {
		return nil, fmt.Errorf("failed to parse csv packets. Error: %w\n", err)
	}
	return messages, err
}
