package algorithms

import (
	"log"

	"github.com/google/btree"
	"github.com/mochaeng/phoenixfl/internal/models"
)

type IpCountItem struct {
	*models.IpCount
}

func (item *IpCountItem) Less(b btree.Item) bool {
	return item.Count > b.(*IpCountItem).Count
}

func UpdateIpCount(tree *btree.BTree, address string) {
	log.Println(address)
	item := &IpCountItem{&models.IpCount{Address: address, Count: 1}}
	existingItem := tree.Get(item)
	if existingItem != nil {
		existingItem.(*IpCountItem).Count++
		// updatedItem := existingItem.(*IpCountItem)
		// tree.Delete(existingItem)
		// updatedItem.Count++
		// tree.ReplaceOrInsert(updatedItem)
	} else {
		tree.ReplaceOrInsert(item)
	}
}
