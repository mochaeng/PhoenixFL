import useWebSocket, { ReadyState } from "react-use-websocket";
import "./App.css";

const WS_URL = "ws://localhost:8080/live-classifications";

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

  const connectionStatus = {
    [ReadyState.CONNECTING]: "Connecting",
    [ReadyState.OPEN]: "Open",
    [ReadyState.CLOSING]: "Closing",
    [ReadyState.CLOSED]: "Closed",
    [ReadyState.UNINSTANTIATED]: "Uninstantiated",
  }[readyState];

  return (
    <>
      <div>{lastMessage?.data}</div>
      <div>{connectionStatus}</div>
    </>
  );
}

export default App;
