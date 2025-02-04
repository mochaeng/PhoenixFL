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

export function IpsTable() {
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
    <div className="flex max-w-[1000px] flex-col justify-center gap-2">
      <p className="mb-4 text-center text-2xl font-bold">
        Network Insights: Malicious Packets Activity
      </p>
      <p className="mb-6 text-center">
        Below are the IP addresses most involved in malicious traffic. This data
        helps identify potential attackers and vulnerable targets within the
        network.
      </p>
      <div className="flex flex-wrap items-center justify-center gap-12">
        {data.map((info) => (
          <div className="w-96">
            <Table key={info.caption}>
              <TableCaption>{info.caption}</TableCaption>
              <TableHeader>
                <TableRow>
                  <TableHead>IP</TableHead>
                  <TableHead>Count</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {info.ips.map((ip, idx) => (
                  <TableRow key={idx}>
                    <TableCell>{ip.key}</TableCell>
                    <TableCell>{ip.value}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ))}
      </div>
      <p className="mt-6 text-center text-sm text-gray-500">
        Note: This data is refreshed periodically based on live packet
        classifications.
      </p>
    </div>
  );
}
