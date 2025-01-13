package stats

import (
	"math"
	"sort"
)

type Metrics struct {
	Mean       float64
	Median     float64
	Quantile75 float64
	Quantile95 float64
	Quantile99 float64
	Max        float64
	Min        float64
}

func ComputeLatenciesMetrics(latencies []float64) *Metrics {
	if len(latencies) == 0 {
		return &Metrics{}
	}

	sort.Float64s(latencies)

	return &Metrics{
		Mean:       mean(latencies),
		Median:     quantile(latencies, 0.5),
		Quantile75: quantile(latencies, 0.75),
		Quantile95: quantile(latencies, 0.95),
		Quantile99: quantile(latencies, 0.99),
		Max:        latencies[len(latencies)-1],
		Min:        latencies[0],
	}
}

func mean(data []float64) float64 {
	sum := 0.0
	for i := 0; i < len(data); i++ {
		sum += data[i]
	}
	return sum / float64(len(data))
}

func quantile(data []float64, q float64) float64 {
	pos := q * float64(len(data)-1)
	lower := int(math.Floor(pos))
	upper := int(math.Ceil(pos))

	if lower == upper {
		return data[lower]
	}
	return data[lower] + (pos-float64(lower))*(data[upper]-data[lower])
}
