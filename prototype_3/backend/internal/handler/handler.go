package handler

import (
	"log"
	"net/http"

	"github.com/gorilla/websocket"
	"github.com/mochaeng/phoenixfl/internal/hub"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

func LiveClassificationsHandler(hub *hub.ClientsHub) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("failed to upgrade websocket. Error: %s\n", err)
			return
		}

		clientID := conn.RemoteAddr().String()
		hub.Add(clientID, conn)
		log.Printf("Client connected: %s\n", clientID)

		defer func() {
			hub.Remove(clientID)
			log.Printf("Client disconnected id [%s]\n", clientID)
		}()

		client, _ := hub.Get(clientID)
		for packet := range client.PacketsChan {
			err := conn.WriteJSON(packet)
			if err != nil {
				log.Printf("could not send message to client. Error: %s\n", err)
				break
			}
		}
	}
}
