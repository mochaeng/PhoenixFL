import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "../ui/separator";
import { formatDuration, formatMaliciousPercentage } from "@/lib/utils";
import { usePacketStore } from "@/store/packet";

export function MainStatsCards() {
  const stats = usePacketStore((state) => state.stats);

  const totalCount = stats.total_packets;
  const totalMalicious = stats.total_malicious;
  const maliciousPercentage = formatMaliciousPercentage(
    100 * (totalMalicious / totalCount),
  );
  const avgLatency = formatDuration(stats.avg_latency);
  const avgClassificationTime = formatDuration(stats.avg_classification_time);

  const data = [
    {
      title: "Total Classified Packets",
      description: "The amount of packets PhoenixFL has already classified",
      content: `${totalCount} packets`,
    },
    {
      title: "Total Malicious Packets",
      description:
        "The amount of packets PhoenixFL has classified as being malicious",
      content: `${totalMalicious} packets`,
    },
    {
      title: "Percentage of Malicious Packets",
      description: "The percentage of malicious classified packets",
      content: `${maliciousPercentage} %`,
    },
    {
      title: "Average Latency Time",
      description: "The time for a classification (network, broker) ",
      content: avgLatency,
    },
    {
      title: "Average Classification Time",
      description:
        "The time a worker took to run classify packet based on federated model",
      content: avgClassificationTime,
    },
  ];

  return (
    <div className="flex flex-wrap justify-center gap-6 p-2">
      {data.map((card) => (
        <Card key={card.title} className="w-full max-w-[330px]">
          <CardHeader className="h-32">
            <CardTitle>{card.title}</CardTitle>
            <CardDescription>{card.description}</CardDescription>
          </CardHeader>
          <Separator />
          <CardContent className="flex items-center justify-center p-2">
            <span className="flex items-center justify-center text-xl">
              {card.content}
            </span>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
