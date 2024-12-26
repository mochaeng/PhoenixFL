package algorithms

import (
	"github.com/google/btree"
	"github.com/mochaeng/phoenixfl/internal/models"
)

type ipCountItem struct {
	*models.IpCount
}

type TopIpTracker struct {
	tree *btree.BTree
	ips  map[string]*ipCountItem
}

func NewTopIpTracker() *TopIpTracker {
	return &TopIpTracker{
		tree: btree.New(32),
		ips:  make(map[string]*ipCountItem),
	}
}

func (item *ipCountItem) Less(b btree.Item) bool {
	return item.Count > b.(*ipCountItem).Count
}

func (t *TopIpTracker) AddOrUpdateIpCount(address string) {
	if existing, found := t.ips[address]; found {
		existing.Count += 1
		t.tree.ReplaceOrInsert(existing)
		// log.Printf("address [%s], count [%d], tree-length [%d]", address, existing.Count, getBTreeLength(t.tree))
	} else {
		item := &ipCountItem{&models.IpCount{Address: address, Count: 1}}
		t.ips[address] = item
		t.tree.ReplaceOrInsert(item)
	}
}

func (t *TopIpTracker) getTopIps(n int) []*ipCountItem {
	var result []*ipCountItem
	t.tree.Ascend(func(item btree.Item) bool {
		if len(result) >= n {
			return false
		}
		result = append(result, item.(*ipCountItem))
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

// func printBTreeLength(tree *btree.BTree) string {
// 	count := 0
// 	tree.Ascend(func(i btree.Item) bool {
// 		count++
// 		return true
// 	})
// 	return count
// }
