package realtime

import (
	"log"
	"net/http"

	"github.com/gorilla/websocket"
	"github.com/mochaeng/phoenixfl/internal/models"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

func StartWebSocketServer(packetsChan <-chan models.ClassifiedPacketResponse) {
	hub := NewClientsHub()
	hub.StartPacketsBroadCastLoop(packetsChan)

	http.HandleFunc("/live-classifications", LiveClassificationsHandler(hub))
	log.Println("server started on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func LiveClassificationsHandler(hub *Hub) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("failed to upgrade websocket. Error: %s\n", err)
			return
		}

		clientID := conn.RemoteAddr().String()
		hub.Add(clientID, conn)
		log.Printf("Client connected: %s\n", clientID)

		client, _ := hub.Get(clientID)

		defer func() {
			client.Hub.Remove(client.ID)
			log.Printf("Client disconnected id [%s]\n", client.ID)
		}()

		for packet := range client.PacketsChan {
			err := client.Conn.WriteJSON(packet)
			if err != nil {
				log.Printf("could not send message to client. Error: %s\n", err)
				break
			}
		}
	}
}
