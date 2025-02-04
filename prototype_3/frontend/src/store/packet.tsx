import { create } from "zustand";
import { PacketResponse, StatsResponse } from "@/lib/responses";

export const SizeThreshold = 8;

export type PacketStore = {
  connectionStatus: string;
  packets: PacketResponse[];
  stats: StatsResponse;
  addPacket: (packet: PacketResponse) => void;
  setConnectionStatus: (status: string) => void;
};

export const usePacketStore = create<PacketStore>()((set) => ({
  connectionStatus: "",
  packets: [],
  stats: {
    total_malicious: 0,
    total_packets: 0,
    avg_classification_time: 0,
    avg_latency: 0,
    malicious_ips: [],
    targeted_ips: [],
    workers_classifications: [],
  },
  addPacket: (packet) =>
    set((state) => {
      const updatedPackets = [packet, ...state.packets];
      if (updatedPackets.length > SizeThreshold) {
        updatedPackets.pop();
      }
      return {
        packets: updatedPackets,
        // totalCount: packet.stats.total_packets,
        // totalMalicious: packet.stats.total_malicious,
        stats: packet.stats,
      };
    }),
  setConnectionStatus: (status) => set({ connectionStatus: status }),
}));
