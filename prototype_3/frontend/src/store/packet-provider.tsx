import { useRef } from "react";
import { createStore, StoreApi } from "zustand";
import { PacketStoreContext } from "./contexts";
import { PacketResponse } from "@/lib/responses";

const SizeThreshold = 8;

export type PacketStore = {
  connectionStatus: string;
  packets: PacketResponse[];
  totalCount: number;
  totalMalicious: number;
  addPacket: (packet: PacketResponse) => void;
  clearPackets: () => void;
  setConnectionStatus: (status: string) => void;
};

export function PacketStoreProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const storeRef = useRef<StoreApi<PacketStore>>();

  if (!storeRef.current) {
    storeRef.current = createStore<PacketStore>()((set) => ({
      packets: [],
      totalCount: 0,
      totalMalicious: 0,
      connectionStatus: "",
      addPacket: (packet) =>
        set((state) => {
          const updatedPackets = [packet, ...state.packets];
          if (updatedPackets.length > SizeThreshold) {
            updatedPackets.pop();
          }
          return {
            packets: updatedPackets,
            totalCount: state.totalCount + 1,
            totalMalicious: packet.is_malicious
              ? state.totalMalicious + 1
              : state.totalMalicious,
          };
        }),
      setConnectionStatus: (status) => set({ connectionStatus: status }),
      clearPackets: () => set({ packets: [] }),
    }));
  }

  return (
    <PacketStoreContext.Provider value={storeRef.current}>
      {children}
    </PacketStoreContext.Provider>
  );
}
