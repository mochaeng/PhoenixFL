package hub

import (
	"log"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/mochaeng/phoenixfl/internal/models"
)

type WebSocketClient struct {
	Conn        *websocket.Conn
	PacketsChan chan models.ClassifiedPacketResponse
}

type ClientsHub struct {
	clients map[string]*WebSocketClient
	mu      sync.RWMutex
}

func NewClientsHub() *ClientsHub {
	return &ClientsHub{
		clients: make(map[string]*WebSocketClient),
	}
}

func (hub *ClientsHub) Add(clientID string, conn *websocket.Conn) {
	hub.mu.Lock()
	defer hub.mu.Unlock()

	_, exists := hub.clients[clientID]
	if exists {
		return
	}

	hub.clients[clientID] = &WebSocketClient{
		Conn:        conn,
		PacketsChan: make(chan models.ClassifiedPacketResponse),
	}
}

func (hub *ClientsHub) Remove(clientID string) {
	hub.mu.Lock()
	defer hub.mu.Unlock()

	client, exists := hub.clients[clientID]
	if !exists {
		return
	}

	client.Conn.Close()
	close(client.PacketsChan)
	delete(hub.clients, clientID)
}

func (hub *ClientsHub) Get(clientID string) (*WebSocketClient, bool) {
	hub.mu.RLock()
	defer hub.mu.RUnlock()
	client, exists := hub.clients[clientID]
	return client, exists
}

func (hub *ClientsHub) BroadcastPacket(packet models.ClassifiedPacketResponse) {
	hub.mu.RLock()
	defer hub.mu.RUnlock()

	for clientID, client := range hub.clients {
		select {
		case client.PacketsChan <- packet:
		default:
			log.Printf("client %s is not able to receive the packet at the moment\n", clientID)
		}
		// err := conn.WriteJSON(message)
		// if err != nil {
		// 	log.Printf("failed to send message to client %v: %v\n", clientID, err)
		// }
	}
}
