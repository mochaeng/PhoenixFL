import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { usePacketStore } from "@/store/packet";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

export function IpsTable({ className }: React.HTMLAttributes<HTMLDivElement>) {
  const stats = usePacketStore((state) => state.stats);

  const data = [
    {
      caption: "IPs that sent most malicious packets",
      ips: stats.malicious_ips,
    },
    {
      caption: "IPs most targeted by malicious packets",
      ips: stats.targeted_ips,
    },
  ];

  return (
    <Card className={cn(className)}>
      <CardHeader>
        <CardTitle>Network Insights: Malicious Packets Activity</CardTitle>
        <CardDescription>
          Below are the IP addresses most involved in malicious traffic. This
          data helps identify potential attackers and vulnerable targets within
          the network.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap items-center justify-center gap-16">
          {data.map((info) => (
            <div className="w-96 rounded-md border">
              <Table key={info.caption}>
                <TableCaption className="pb-2">{info.caption}</TableCaption>
                <TableHeader>
                  <TableRow>
                    <TableHead>IP</TableHead>
                    <TableHead>Count</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {info.ips.map((ip, idx) => (
                    <TableRow key={ip.value * idx}>
                      <TableCell>{ip.key}</TableCell>
                      <TableCell>{ip.value}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
