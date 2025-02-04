package models

import "github.com/google/btree"

type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

type ItemCount[V Numeric] struct {
	Key   string `json:"key"`
	Value V      `json:"value"`
}

func (item *ItemCount[V]) Less(other btree.Item) bool {
	otherItem := other.(*ItemCount[V])
	if item.Value != otherItem.Value {
		return item.Value < otherItem.Value
	}
	return item.Key < otherItem.Key
}
