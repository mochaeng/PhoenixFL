import { usePacketStore } from "@/hooks/use-packet-store";

export function WebsocketStatus() {
  const connectionStatus = usePacketStore((state) => state.connectionStatus);

  return (
    <div>
      <span className="font-semibold">Websocket connection</span>:{" "}
      {connectionStatus}
    </div>
  );
}
