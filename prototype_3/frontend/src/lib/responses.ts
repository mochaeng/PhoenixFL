import { z } from "zod";

const metadataSchema = z.object({
  ipv4_src_addr: z.string(),
  l4_src_port: z.number(),
  ipv4_dst_addr: z.string(),
  l4_dst_port: z.number(),
});

const statsSchema = z.object({
  total_packets: z.number(),
  total_malicious: z.number(),
});

export const PacketSchema = z.object({
  id: z.string(),
  metadata: metadataSchema,
  stats: statsSchema,
  classification_time: z.number(),
  total_time: z.number(),
  worker_name: z.string(),
  is_malicious: z.boolean(),
});
export type PacketResponse = z.infer<typeof PacketSchema>;
