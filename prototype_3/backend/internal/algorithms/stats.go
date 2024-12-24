package algorithms

import (
	"github.com/google/btree"
	"github.com/mochaeng/phoenixfl/internal/models"
)

type PacketStats struct {
	TotalPackets    int64
	TotalMalicious  int64
	SourceCount     map[string]int64
	DestCount       map[string]int64
	WorkerCount     map[string]int64
	AvgLatency      float64
	sourceCountTree *btree.BTree
	destCountTree   *btree.BTree
	// SourcePriorityQueue PriorityQueue
	// DestPriorityQueue   PriorityQueue
}

func NewPacketStats() *PacketStats {
	packetStats := PacketStats{
		SourceCount:     make(map[string]int64),
		DestCount:       make(map[string]int64),
		sourceCountTree: btree.New(2),
		destCountTree:   btree.New(2),
		// SourcePriorityQueue: make(PriorityQueue, 0),
		// DestPriorityQueue:   make(PriorityQueue, 0),
	}
	// heap.Init(&packetStats.SourcePriorityQueue)
	// heap.Init(&packetStats.DestPriorityQueue)
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

		UpdateIpCount(stats.sourceCountTree, sourceIp)
		UpdateIpCount(stats.destCountTree, destIp)

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

func (stats *PacketStats) getTopIps(tree *btree.BTree, n int) []*IpCountItem {
	var topIPs []*IpCountItem
	tree.Ascend(func(item btree.Item) bool {
		if len(topIPs) >= n {
			return false
		}
		topIPs = append(topIPs, item.(*IpCountItem))
		return true
	})
	return topIPs
}

func (stats *PacketStats) GetTopMaliciousIps(n int) []*IpCountItem {
	return stats.getTopIps(stats.sourceCountTree, n)
}
