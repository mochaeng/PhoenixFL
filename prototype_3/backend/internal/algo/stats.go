package algo

import "github.com/mochaeng/phoenixfl/internal/models"

type PacketStats struct {
	TotalPackets          int64
	TotalMalicious        int64
	SourceCount           map[string]int64
	DestCount             map[string]int64
	WorkerCount           map[string]int64
	AvgLatency            float64
	AvgClassificationTime float64

	maliciousIps Tracker[int]
	targetedIps  Tracker[int]
}

func NewPacketStats() *PacketStats {
	packetStats := PacketStats{
		SourceCount:  make(map[string]int64),
		DestCount:    make(map[string]int64),
		WorkerCount:  make(map[string]int64),
		maliciousIps: NewTrackerBTree[int](),
		targetedIps:  NewTrackerBTree[int](),
	}
	return &packetStats
}

func (stats *PacketStats) Update(packet models.Packet) {
	stats.TotalPackets += 1
	stats.AvgLatency += (packet.Latency - stats.AvgLatency) / float64(stats.TotalPackets)
	stats.AvgClassificationTime += (packet.ClassificationTime - stats.AvgClassificationTime) / float64(stats.TotalPackets)

	if packet.IsMalicious {
		stats.TotalMalicious += 1

		sourceIp := packet.Metadata.SourceIpAddr
		destIp := packet.Metadata.DestIpAddr

		stats.SourceCount[sourceIp]++
		stats.DestCount[destIp]++

		stats.maliciousIps.AddOrUpdateItemCount(sourceIp)
		stats.targetedIps.AddOrUpdateItemCount(destIp)
	}
}

func (stats *PacketStats) GetStatsResponse(ipsAmount int) *models.StatsResponse {
	maliciousIPs := stats.GetTopMaliciousIps(ipsAmount)
	targetedIps := stats.GetTopTargetedIps(ipsAmount)
	return &models.StatsResponse{
		TotalPackets:          stats.TotalPackets,
		TotalMalicious:        stats.TotalMalicious,
		MaliciousIps:          maliciousIPs,
		TargetedIps:           targetedIps,
		AvgLatency:            stats.AvgLatency,
		AvgClassificationTime: stats.AvgClassificationTime,
	}
}

func (stats *PacketStats) GetTopMaliciousIps(n int) []*models.ItemCount[int] {
	return stats.maliciousIps.GetTopItems(n)
}

func (stats *PacketStats) GetTopTargetedIps(n int) []*models.ItemCount[int] {
	return stats.targetedIps.GetTopItems(n)
}
