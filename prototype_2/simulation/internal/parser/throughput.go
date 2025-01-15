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

type SimulationRecord struct {
	Timer float64 `json:"mean"`
}

type WorkerRecord struct {
	ProcessedPackets uint64  `json:"processed_packets"`
	Timer            float64 `json:"total_timer"`
}

type FileFilter struct {
	Prefix    string
	Extension string
}

type FileProcessor[T any] func(data []byte) (T, error)

func walkAndProcess[T any](root string, filter FileFilter, processor FileProcessor[T]) ([]T, error) {
	var results []T

	err := filepath.Walk(root, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("could not access path [%s]. Error: %w\n", path, err)
		}

		if !info.IsDir() &&
			strings.HasPrefix(filepath.Base(path), filter.Prefix) &&
			filepath.Ext(path) == filter.Extension {

			data, err := os.ReadFile(path)
			if err != nil {
				return fmt.Errorf("could not read file [%s]. Error: %w\n", path, err)
			}

			result, err := processor(data)
			if err != nil {
				return fmt.Errorf("could not process file [%s]. Error: %w", path, err)
			}

			results = append(results, result)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	return results, nil
}

func GetThroughputSimulationResult(root string, numWorkers int) (*models.SimulationThroughputResult, error) {
	filter := FileFilter{
		Prefix:    fmt.Sprintf("%d-", numWorkers),
		Extension: ".json",
	}

	processor := func(data []byte) (SimulationRecord, error) {
		var record SimulationRecord
		if err := json.Unmarshal(data, &record); err != nil {
			return SimulationRecord{}, fmt.Errorf("could not marshal file. Error: %w\n", err)
		}
		return record, nil
	}

	records, err := walkAndProcess(root, filter, processor)
	if err != nil {
		return nil, err
	}

	timers := make([]float64, len(records))
	for i, record := range records {
		timers[i] = record.Timer
	}

	metrics := stats.ComputeMetrics(timers)
	return &models.SimulationThroughputResult{
		Timers:     timers,
		Mean:       metrics.Mean,
		Median:     metrics.Median,
		Quantile75: metrics.Quantile75,
		Quantile95: metrics.Quantile95,
		Quantile99: metrics.Quantile99,
	}, nil
}

func AggregateWorkerThroughputData(root string) (*models.WorkerThroughputRecords, error) {
	filter := FileFilter{
		Prefix:    "worker",
		Extension: ".json",
	}

	processor := func(data []byte) (WorkerRecord, error) {
		var record WorkerRecord
		if err := json.Unmarshal(data, &record); err != nil {
			return WorkerRecord{}, fmt.Errorf("could not unmarshal file. Error: %w\n", err)
		}
		return record, nil
	}

	records, err := walkAndProcess(root, filter, processor)
	if err != nil {
		return nil, err
	}

	result := &models.WorkerThroughputRecords{
		Timers: make([]float64, len(records)),
	}

	for i, record := range records {
		result.Timers[i] = record.Timer
		result.TotalProcessedPackets += record.ProcessedPackets
		result.TotalFiles++
	}

	metrics := stats.ComputeMetrics(result.Timers)
	result.Mean = metrics.Mean

	return result, nil
}
