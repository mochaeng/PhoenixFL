// import { PacketStoreContext } from "@/store/contexts";
// import { PacketStore } from "@/store/packet-provider";
// import { useContext } from "react";
// import { useStore } from "zustand";

// export const usePacketStore = <T,>(selector: (state: PacketStore) => T) => {
//   const store = useContext(PacketStoreContext);
//   if (!store) {
//     throw new Error("usePacketStore must be used within a PacketStoreProvider");
//   }
//   return useStore(store, selector);
// };
