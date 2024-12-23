import { usePacketStore } from "@/store/packet";
import React from "react";

export function WebsocketStatus({
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  const connectionStatus = usePacketStore((state) => state.connectionStatus);

  return (
    <div {...props}>
      <span className="font-semibold">PhoenixFL status</span>:{" "}
      {connectionStatus}
    </div>
  );
}
