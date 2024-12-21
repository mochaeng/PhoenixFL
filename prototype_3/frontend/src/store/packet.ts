import { PacketResponse } from "@/lib/responses";
import { create } from "zustand";

const SizeThreshold = 8;

type PacketStore = {
  packets: PacketResponse[];
  addPacket: (packet: PacketResponse) => void;
  clearPackets: () => void;
};

export const usePacketStore = create<PacketStore>()((set) => ({
  packets: [],
  addPacket: (packet) =>
    set((state) => {
      const updatedPackets = [...state.packets, packet];
      if (updatedPackets.length > SizeThreshold) {
        updatedPackets.shift();
      }
      return {
        packets: updatedPackets,
      };
    }),
  clearPackets: () => set({ packets: [] }),
}));
