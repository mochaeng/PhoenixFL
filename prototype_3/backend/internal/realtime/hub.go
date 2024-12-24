package realtime

import (
	"log"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/mochaeng/phoenixfl/internal/models"
)

type Hub struct {
	clients map[string]*Client
	mu      sync.RWMutex
}

func NewClientsHub() *Hub {
	return &Hub{
		clients: make(map[string]*Client),
	}
}

func (hub *Hub) Add(clientID string, conn *websocket.Conn) {
	hub.mu.Lock()
	defer hub.mu.Unlock()

	_, exists := hub.clients[clientID]
	if exists {
		return
	}

	hub.clients[clientID] = &Client{
		ID:          clientID,
		Conn:        conn,
		Hub:         hub,
		PacketsChan: make(chan models.ClassifiedPacketResponse),
	}
}

func (hub *Hub) Remove(clientID string) {
	hub.mu.Lock()
	defer hub.mu.Unlock()

	client, exists := hub.clients[clientID]
	if !exists {
		return
	}

	close(client.PacketsChan)
	client.Conn.Close()
	delete(hub.clients, clientID)
}

func (hub *Hub) Get(clientID string) (*Client, bool) {
	hub.mu.RLock()
	defer hub.mu.RUnlock()
	client, exists := hub.clients[clientID]
	return client, exists
}

func (hub *Hub) broadcastPacket(packet models.ClassifiedPacketResponse) {
	hub.mu.RLock()
	defer hub.mu.RUnlock()

	for clientID, client := range hub.clients {
		select {
		case client.PacketsChan <- packet:
		default:
			log.Printf("client %s is not able to receive the packet at the moment\n", clientID)
		}
	}
}

func (hub *Hub) StartPacketsBroadCastLoop(packetsChan <-chan models.ClassifiedPacketResponse) {
	go func() {
		for packet := range packetsChan {
			hub.broadcastPacket(packet)
		}
	}()
}
