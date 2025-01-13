package mb

import (
	"log"
	"sync"
	"testing"
	"time"
)

func TestConcurrentQueue_ConcurrentEnqueueDequeue(t *testing.T) {
	t.Run("multiple goroutines enqueuing and dequeuing", func(t *testing.T) {
		queue := NewConcurrentQueue[int]()
		numItems := 1000
		numGoroutines := 10

		var enqueued, dequeued sync.Map
		var wg sync.WaitGroup

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				log.Println("Publishing...")
				for j := 0; j < numItems/numGoroutines; j++ {
					item := workerID*numItems + j
					queue.Enqueue(item)
					enqueued.Store(item, true)
				}
			}(i)
		}

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				log.Println("Consuming...")
				for {
					if item, ok := queue.Dequeue(); ok {
						dequeued.Store(item, true)
					} else if queue.Len() == 0 {
						time.Sleep(time.Millisecond)
						if queue.Len() == 0 {
							break
						}
					}
				}
			}()
		}

		wg.Wait()

		enqueuedCount := 0
		enqueued.Range(func(key any, value any) bool {
			enqueuedCount++
			return true
		})

		dequeuedCount := 0
		dequeued.Range(func(key any, value any) bool {
			dequeuedCount++
			return true
		})

		if enqueuedCount != numItems {
			t.Errorf("Expected %d enqueued items, got %d", numItems, enqueuedCount)
		}
		if dequeuedCount != numItems {
			t.Errorf("Expected %d dequeued items, got %d", numItems, enqueuedCount)
		}
	})
}

func TestConcurrentQueue_EmptyDequeue(t *testing.T) {
	queue := NewConcurrentQueue[int]()

	if _, ok := queue.Dequeue(); ok {
		t.Error("Expected Dequeue on empty queue to return false")
	}

	queue.Enqueue(1)
	if item, ok := queue.Dequeue(); !ok || item != 1 {
		t.Error("Expected Dequeue to return 1")
	}

	if _, ok := queue.Dequeue(); ok {
		t.Error("Expected Dequeue on empty queue to return false")
	}
}

func TestConcurrentQueue_OrderPreservation(t *testing.T) {
	queue := NewConcurrentQueue[int]()
	items := []int{1, 2, 3, 4, 5}

	for _, item := range items {
		queue.Enqueue(item)
	}

	for _, expected := range items {
		if item, ok := queue.Dequeue(); !ok || item != expected {
			t.Errorf("Expected %d, got %d", expected, item)
		}
	}
}

// /run test with [-race] flag on
func TestConcurrentQueue_RaceCoditions(t *testing.T) {
	queue := NewConcurrentQueue[int]()
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(3)

		go func(val int) {
			defer wg.Done()
			queue.Enqueue(val)
		}(i)

		go func() {
			defer wg.Done()
			queue.Dequeue()
		}()

		go func() {
			defer wg.Done()
			_ = queue.Len()
		}()
	}

	wg.Wait()
}

func TestConcurrentQueue_WaitForItem(t *testing.T) {
	queue := NewConcurrentQueue[int]()

	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		queue.WaitForItem()
	}()

	time.Sleep(100 * time.Millisecond)

	queue.Enqueue(69)

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Error("WaitForItem did not return after item was enqueued")
	}
}

func TestConcurrentQueue_MultipleWaiters(t *testing.T) {
	queue := NewConcurrentQueue[int]()
	numWaiters := 3

	var wg sync.WaitGroup
	wg.Add(numWaiters)

	for i := 0; i < numWaiters; i++ {
		go func() {
			defer wg.Done()
			queue.WaitForItem()
		}()
	}

	time.Sleep(100 * time.Millisecond)

	queue.Enqueue(69)

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Error("Not all waiters were notified")
	}
}
