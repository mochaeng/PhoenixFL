import { PacketResponse } from "@/lib/responses";
import { create } from "zustand";

type PacketStore = {
  packets: PacketResponse[];
  addPacket: (packet: PacketResponse) => void;
  clearPackets: () => void;
};

export const usePacketStore = create<PacketStore>((set) => ({
  packets: [],
  addPacket: (packet) =>
    set((state) => {
      const updatedPackets = state.packets.slice();
      updatedPackets.push(packet);
      if (updatedPackets.length > 8) {
        updatedPackets.shift();
      }
      return {
        packets: updatedPackets,
      };
    }),
  clearPackets: () => set({ packets: [] }),
}));
