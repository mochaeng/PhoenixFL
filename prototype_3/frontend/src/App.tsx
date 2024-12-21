import "./App.css";
import {
  createBrowserRouter,
  Navigate,
  RouterProvider,
} from "react-router-dom";
import ErrorPage from "./pages/error-page";
import RootLayout from "./pages/root-layout";

// const WS_URL = "ws://localhost:8080/live-classifications";

const router = createBrowserRouter([
  {
    path: "/",
    element: <RootLayout />,
    errorElement: <ErrorPage />,
    children: [
      { index: true, element: <Navigate to="/dashboard/home" replace /> },
      {
        path: "dashboard",
        // element: <DashboardLayout />,
        children: [
          { index: true, element: <Navigate to="home" replace /> },
          { path: "home", element: <div>Home</div> },
          { path: "overview", element: <div>Overview</div> },
          { path: "recent", element: <div>Recent</div> },
          { path: "performance", element: <div>Performance</div> },
        ],
      },
      {
        path: "history",
        children: [
          { index: true, element: <Navigate to="search" replace /> },
          { path: "search", element: <div>Search</div> },
        ],
      },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
  // const { lastMessage, readyState } = useWebSocket(WS_URL, {
  //   shouldReconnect: (closedEvent) => {
  //     const { code } = closedEvent;
  //     console.log("websocket connection was closed with code: ", code);
  //     return true;
  //   },
  //   reconnectAttempts: 10,
  //   reconnectInterval: (attemptNumber) =>
  //     Math.min(Math.pow(2, attemptNumber) * 1000, 10000),
  // });

  // const packets = usePacketStore((state) => state.packets);
  // const addPacket = usePacketStore((state) => state.addPacket);

  // useEffect(() => {
  //   if (lastMessage) {
  //     const data = JSON.parse(lastMessage.data);
  //     const parsed = PacketSchema.safeParse(data);
  //     if (!parsed.success) {
  //       console.log(
  //         "could not parse response from webscoket. Error: ",
  //         parsed.error,
  //       );
  //       return;
  //     }
  //     addPacket(parsed.data);
  //   }
  // }, [lastMessage, addPacket]);

  // const connectionStatus = {
  //   [ReadyState.CONNECTING]: "Connecting",
  //   [ReadyState.OPEN]: "Open",
  //   [ReadyState.CLOSING]: "Closing",
  //   [ReadyState.CLOSED]: "Closed",
  //   [ReadyState.UNINSTANTIATED]: "Uninstantiated",
  // }[readyState];

  // return (
  //   <div>
  //     <h2>Last packets (100)</h2>
  //     <a> wtrf</a>
  //     <div>{connectionStatus}</div>
  //     <div>
  //       {packets.map((packet) => (
  //         <li key={packet.id} className="list-none p-2">
  //           {JSON.stringify(packet)}
  //         </li>
  //       ))}
  //     </div>
  //   </div>
  // );
}

export default App;
