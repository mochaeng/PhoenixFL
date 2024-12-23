import { createContext } from "react";
import { StoreApi } from "zustand";
import { PacketStore } from "./packet-provider";

export const PacketStoreContext = createContext<StoreApi<PacketStore> | null>(
  null,
);
