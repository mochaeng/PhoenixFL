import useWebSocket, { ReadyState } from "react-use-websocket";
import "./App.css";
import { useEffect, useState } from "react";
import { PacketSchema } from "./lib/responses";
import { createBrowserRouter } from "react-router-dom";
import { Root } from "node_modules/@radix-ui/react-slot/dist";
import ErrorPage from "./pages/error-page";
import { usePacketStore } from "./store/packet";

const WS_URL = "ws://localhost:8080/live-classifications";

// const router = createBrowserRouter([
//   {
//     path: "/",
//     element: <Root />,
//     errorElement: <ErrorPage />,
//     children: [
//       // { index: true, element: <HomePage /> },
//       // { path: ":username", element: <ProfilePage /> },
//       // { path: "posts" },
//     ],
//   },
// ]);

function App() {
  const { lastMessage, readyState } = useWebSocket(WS_URL, {
    shouldReconnect: (closedEvent) => {
      const { code } = closedEvent;
      console.log("websocket connection was closed with code: ", code);
      return true;
    },
    reconnectAttempts: 10,
    reconnectInterval: (attemptNumber) =>
      Math.min(Math.pow(2, attemptNumber) * 1000, 10000),
  });

  const { packets, addPacket } = usePacketStore();

  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data);
      const parsed = PacketSchema.safeParse(data);
      if (!parsed.success) {
        console.log(
          "could not parse response from webscoket. Error: ",
          parsed.error,
        );
        return;
      }
      addPacket(parsed.data);
    }
  }, [lastMessage, addPacket]);

  const connectionStatus = {
    [ReadyState.CONNECTING]: "Connecting",
    [ReadyState.OPEN]: "Open",
    [ReadyState.CLOSING]: "Closing",
    [ReadyState.CLOSED]: "Closed",
    [ReadyState.UNINSTANTIATED]: "Uninstantiated",
  }[readyState];

  return (
    <div>
      <h2>Last packets (100)</h2>
      <a> wtrf</a>
      <div>{connectionStatus}</div>
      <div>
        {packets.map((packet) => (
          <li key={packet.id} className="list-none p-2">
            {JSON.stringify(packet)}
          </li>
        ))}
      </div>
    </div>
  );
}

export default App;
