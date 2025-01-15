package stats

import (
	"math"
	"sort"
)

type Metrics struct {
	Mean       float64 `json:"mean"`
	Median     float64 `json:"median"`
	Quantile75 float64 `json:"quantile_75"`
	Quantile95 float64 `json:"quantile_95"`
	Quantile99 float64 `json:"quantile_99"`
	Max        float64 `json:"max"`
	Min        float64 `json:"min"`
}

func ComputeMetrics(data []float64) *Metrics {
	if len(data) == 0 {
		return &Metrics{}
	}

	sort.Float64s(data)

	return &Metrics{
		Mean:       mean(data),
		Median:     quantile(data, 0.5),
		Quantile75: quantile(data, 0.75),
		Quantile95: quantile(data, 0.95),
		Quantile99: quantile(data, 0.99),
		Max:        data[len(data)-1],
		Min:        data[0],
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
