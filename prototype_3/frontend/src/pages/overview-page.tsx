import { LatestPacketsTable } from "@/components/home/latest-packets-table";
import { PacketSchema } from "@/lib/responses";
import { usePacketStore } from "@/store/packet";
import { useEffect } from "react";
import useWebSocket, { ReadyState } from "react-use-websocket";

const WS_URL = "ws://localhost:8080/live-classifications";

export function OverviewPage() {
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

  const addPacket = usePacketStore((state) => state.addPacket);

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
    <div className="max-w-screen-xl flex flex-col gap-2">
      <div>
        <span className="font-semibold">Websocket connection</span>:{" "}
        {connectionStatus}
      </div>
      <LatestPacketsTable />
    </div>
  );
}
