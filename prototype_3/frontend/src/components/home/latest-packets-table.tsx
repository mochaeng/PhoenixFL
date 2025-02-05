import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn, formatSeconds } from "@/lib/utils";
import { usePacketStore } from "@/store/packet";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function LatestPacketsTable({
  className,
}: React.HTMLAttributes<HTMLDivElement>) {
  const packets = usePacketStore((state) => state.packets);

  return (
    <Card className={cn(className)}>
      <CardHeader>
        <CardTitle>Recent Packet Classifications</CardTitle>
        <CardDescription>
          The latest classification results. It shows the IPs sources and ports.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div>
          <Table>
            <TableCaption>
              (Red values indicate a malicious packet)
            </TableCaption>
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
                <TableRow
                  className={cn({
                    "text-red-600": packet.packet_info.is_malicious,
                  })}
                  key={packet.id}
                >
                  <TableCell>
                    {packet.packet_info.metadata.ipv4_src_addr}
                  </TableCell>
                  <TableCell>
                    {packet.packet_info.metadata.l4_src_port}
                  </TableCell>
                  <TableCell>
                    {packet.packet_info.metadata.ipv4_dst_addr}
                  </TableCell>
                  <TableCell>
                    {packet.packet_info.metadata.l4_dst_port}
                  </TableCell>
                  <TableCell>
                    {formatSeconds(packet.packet_info.latency)}
                  </TableCell>
                  <TableCell>
                    {packet.packet_info.is_malicious ? "True" : "False"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
