import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { usePacketStore } from "@/hooks/use-packet-store";
import { formatSeconds } from "@/lib/utils";

export function LatestPacketsTable() {
  const packets = usePacketStore((state) => state.packets);

  return (
    <Table>
      <TableCaption>The most recent packets</TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead>IP source</TableHead>
          <TableHead>Port source</TableHead>
          <TableHead>IP dest</TableHead>
          <TableHead>Port dest</TableHead>
          <TableHead>Total Time (Seconds)</TableHead>
          <TableHead>Classification</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {packets.map((packet) => (
          <TableRow key={packet.id}>
            <TableCell>{packet.metadata.ipv4_src_addr}</TableCell>
            <TableCell>{packet.metadata.l4_src_port}</TableCell>
            <TableCell>{packet.metadata.ipv4_dst_addr}</TableCell>
            <TableCell>{packet.metadata.l4_dst_port}</TableCell>
            <TableCell>{formatSeconds(packet.total_time)}</TableCell>
            <TableCell>{packet.is_malicious ? "True" : "False"}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
