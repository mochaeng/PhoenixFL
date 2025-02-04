package algo

import (
	"fmt"
	"strings"

	"github.com/google/btree"
	"github.com/mochaeng/phoenixfl/internal/models"
)

type Tracker[V models.Numeric] interface {
	AddOrUpdateItemCount(key string)
	GetTopItems(n int) []*models.ItemCount[V]
}

type TrackerBTree[V models.Numeric] struct {
	tree  *btree.BTree
	items map[string]*models.ItemCount[V]
}

func NewTrackerBTree[V models.Numeric]() *TrackerBTree[V] {
	return &TrackerBTree[V]{
		tree:  btree.New(32),
		items: make(map[string]*models.ItemCount[V]),
	}
}

func (tracker *TrackerBTree[V]) AddOrUpdateItemCount(key string) {
	if existing, found := tracker.items[key]; found {
		tracker.tree.Delete(existing)
		existing.Value += 1
		tracker.tree.ReplaceOrInsert(existing)
		// log.Println(getBTreeRepresentation(tracker.tree))
	} else {
		item := &models.ItemCount[V]{Key: key, Value: 1}
		tracker.items[key] = item
		tracker.tree.ReplaceOrInsert(item)
	}
}

func (tracker *TrackerBTree[V]) GetTopItems(n int) []*models.ItemCount[V] {
	var result []*models.ItemCount[V]
	tracker.tree.Descend(func(item btree.Item) bool {
		if len(result) >= n {
			return false
		}
		result = append(result, item.(*models.ItemCount[V]))
		return true
	})
	return result
}

func (tracker *TrackerBTree[V]) getBTreeRepresentation(tree *btree.BTree) string {
	var build strings.Builder
	fmt.Fprintf(&build, "[")
	tree.Descend(func(item btree.Item) bool {
		itemCount := item.(*models.ItemCount[V])
		fmt.Fprintf(&build, "{%s %v}", itemCount.Key, itemCount.Value)
		return true
	})
	fmt.Fprintf(&build, "]")
	return build.String()
}

func getBTreeLength(tree *btree.BTree) int {
	count := 0
	tree.Ascend(func(i btree.Item) bool {
		count++
		return true
	})
	return count
}
