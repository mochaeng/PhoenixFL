package algorithms

import (
	"fmt"
	"strings"

	"github.com/google/btree"
	"github.com/mochaeng/phoenixfl/internal/models"
)

type ipCountItem struct {
	*models.IpCount
}

func (item *ipCountItem) Less(b btree.Item) bool {
	new := b.(*ipCountItem)
	if item.Count != new.Count {
		return item.Count < new.Count
	}
	return item.Address < new.Address
}

type IpTrackerBTree struct {
	tree *btree.BTree
	ips  map[string]*ipCountItem
}

func NewIpTrackerBTree() *IpTrackerBTree {
	return &IpTrackerBTree{
		tree: btree.New(32),
		ips:  make(map[string]*ipCountItem),
	}
}

func (tracker *IpTrackerBTree) AddOrUpdateIpCount(address string) {
	if existing, found := tracker.ips[address]; found {
		tracker.tree.Delete(existing)
		existing.Count += 1
		tracker.tree.ReplaceOrInsert(existing)
		// log.Println(getBTreeRepresentation(tracker.tree))
	} else {
		item := &ipCountItem{&models.IpCount{Address: address, Count: 1}}
		tracker.ips[address] = item
		tracker.tree.ReplaceOrInsert(item)
	}
}

func (tracker *IpTrackerBTree) GetTopIps(n int) []*models.IpCount {
	var result []*models.IpCount
	tracker.tree.Descend(func(item btree.Item) bool {
		if len(result) >= n {
			return false
		}
		result = append(result, item.(*ipCountItem).IpCount)
		return true
	})
	return result
}

func getBTreeLength(tree *btree.BTree) int {
	count := 0
	tree.Ascend(func(i btree.Item) bool {
		count++
		return true
	})
	return count
}

func getBTreeRepresentation(tree *btree.BTree) string {
	var build strings.Builder
	fmt.Fprintf(&build, "[")
	tree.Descend(func(item btree.Item) bool {
		ip := item.(*ipCountItem)
		fmt.Fprintf(&build, "{%s %d}", ip.Address, ip.Count)
		return true
	})
	fmt.Fprintf(&build, "]")
	return build.String()
}
