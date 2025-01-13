package mb

import "sync"

type ConcurrentQueue[T any] struct {
	queue []T

	sync.Mutex
	cond *sync.Cond
}

func NewConcurrentQueue[T any]() *ConcurrentQueue[T] {
	q := &ConcurrentQueue[T]{
		queue: make([]T, 0),
	}
	q.cond = sync.NewCond(&q.Mutex)
	return q
}

func (q *ConcurrentQueue[T]) Enqueue(item T) {
	q.Lock()
	q.queue = append(q.queue, item)
	q.cond.Broadcast()
	q.Unlock()
}

func (q *ConcurrentQueue[T]) Dequeue() (T, bool) {
	q.Lock()
	defer q.Unlock()

	var zero T
	if len(q.queue) == 0 {
		return zero, false
	}

	item := q.queue[0]
	q.queue = q.queue[1:]
	return item, true
}

func (q *ConcurrentQueue[T]) Len() int {
	q.Lock()
	defer q.Unlock()
	return len(q.queue)
}

func (q *ConcurrentQueue[T]) WaitForItem() {
	q.Lock()
	defer q.Unlock()

	for len(q.queue) == 0 {
		q.cond.Wait()
	}
}
