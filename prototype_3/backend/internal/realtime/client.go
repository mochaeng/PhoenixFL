package realtime

import (
	"github.com/gorilla/websocket"
	"github.com/mochaeng/phoenixfl/internal/models"
)

type Client struct {
	ID          string
	Hub         *Hub
	Conn        *websocket.Conn
	PacketsChan chan models.ClassifiedPacketResponse
}
