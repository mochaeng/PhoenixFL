import { z } from "zod";

const metadataSchema = z.object({
  ipv4_src_addr: z.string(),
  l4_src_port: z.number(),
  ipv4_dst_addr: z.string(),
  l4_dst_port: z.number(),
});

const itemCountSchema = z.object({
  key: z.string(),
  value: z.number(),
});

const statsSchema = z.object({
  total_packets: z.number(),
  total_malicious: z.number(),
  malicious_ips: z.array(itemCountSchema),
  targeted_ips: z.array(itemCountSchema),
  workers_classifications: z.array(itemCountSchema),
  avg_latency: z.number(),
  avg_classification_time: z.number(),
});
export type StatsResponse = z.infer<typeof statsSchema>;

const packetInfoSchema = z.object({
  metadata: metadataSchema,
  classification_time: z.number(),
  latency: z.number(),
  is_malicious: z.boolean(),
  worker_name: z.string(),
  timestamp: z.number(),
});

export const PacketSchema = z.object({
  id: z.string(),
  stats: statsSchema,
  packet_info: packetInfoSchema,
});
export type PacketResponse = z.infer<typeof PacketSchema>;
