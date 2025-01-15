package parser

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/mochaeng/phoenix-detector/internal/models"
	"github.com/mochaeng/phoenix-detector/internal/stats"
)

func GetAllThroughputsAggregatedRecords(root string) (*models.AggregatedThroughput, error) {
	aggregatedThroughput := &models.AggregatedThroughput{
		Timers: make([]float64, 0),
	}

	err := filepath.Walk(root, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("could not access path [%s]. Error: %w\n", path, err)
		}

		startsWith := strings.HasPrefix(filepath.Base(path), "worker")

		if !info.IsDir() && startsWith && filepath.Ext(path) == ".json" {
			data, err := os.ReadFile(path)
			if err != nil {
				return fmt.Errorf("could not read file [%s]. Error: %w\n", path, err)
			}

			var record models.RecordThroughput
			if err := json.Unmarshal(data, &record); err != nil {
				return fmt.Errorf("could not unmarshal file. Error %w\n", err)
			}

			aggregatedThroughput.TotalFiles++
			aggregatedThroughput.Timers = append(aggregatedThroughput.Timers, record.Timer)
			aggregatedThroughput.TotalProcessedPackets += record.ProcessedPackets
		}

		return nil
	})
	if err != nil {
		return nil, err
	}

	_ = stats.ComputeMetrics(aggregatedThroughput.Timers)

	return aggregatedThroughput, nil
}
