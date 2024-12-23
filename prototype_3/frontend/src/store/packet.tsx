import { create } from "zustand";
import { PacketResponse } from "@/lib/responses";

export const SizeThreshold = 8;

export type PacketStore = {
  connectionStatus: string;
  packets: PacketResponse[];
  totalCount: number;
  totalMalicious: number;
  addPacket: (packet: PacketResponse) => void;
  setConnectionStatus: (status: string) => void;
};

export const usePacketStore = create<PacketStore>()((set) => ({
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
        totalCount: packet.stats.total_packets,
        totalMalicious: packet.stats.total_malicious,
      };
    }),
  setConnectionStatus: (status) => set({ connectionStatus: status }),
}));
