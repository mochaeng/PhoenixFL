package algorithms

import (
	"github.com/mochaeng/phoenixfl/internal/models"
)

type PacketStats struct {
	TotalPackets   int64
	TotalMalicious int64
	SourceCount    map[string]int64
	DestCount      map[string]int64
	WorkerCount    map[string]int64
	AvgLatency     float64

	topMaliciousIps *TopIpTracker
}

func NewPacketStats() *PacketStats {
	packetStats := PacketStats{
		SourceCount:     make(map[string]int64),
		DestCount:       make(map[string]int64),
		topMaliciousIps: NewTopIpTracker(),
	}
	return &packetStats
}

func (stats *PacketStats) Update(packet models.Packet) {
	stats.TotalPackets += 1
	stats.AvgLatency += (packet.Latency - stats.AvgLatency) / float64(stats.TotalPackets)

	if packet.IsMalicious {
		stats.TotalMalicious += 1

		sourceIp := packet.Metadata.SourceIpAddr
		destIp := packet.Metadata.DestIpAddr

		stats.SourceCount[sourceIp]++
		stats.DestCount[destIp]++

		stats.topMaliciousIps.AddOrUpdateIpCount(sourceIp)

		// AddOrUpdateIpCount(stats.sourceCountTree, sourceIp)
		// AddOrUpdateIpCount(stats.destCountTree, destIp)

		// heap.Push(&stats.SourcePriorityQueue, &Item{
		// 	Value:    sourceIp,
		// 	Priority: stats.SourceCount[sourceIp],
		// })
		// heap.Push(&stats.DestPriorityQueue, &Item{
		// 	Value:    destIp,
		// 	Priority: stats.DestCount[destIp],
		// })
	}
}

func (stats *PacketStats) GetTopMaliciousIps(n int) []*ipCountItem {
	return stats.topMaliciousIps.getTopIps(n)
}
