import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "../ui/separator";
import { formatMaliciousPercentage } from "@/lib/utils";
import { usePacketStore } from "@/store/packet";

export function MainStatsCards() {
  const totalCount = usePacketStore((state) => state.totalCount);
  const totalMalicious = usePacketStore((state) => state.totalMalicious);
  const maliciousPercentage = formatMaliciousPercentage(
    totalMalicious / totalCount,
  );

  console.log(totalCount);

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
  ];

  return (
    <div className="flex flex-wrap justify-center gap-4">
      {data.map((card) => (
        <Card
          key={card.title}
          className="aspect-video w-full min-w-[100px] max-w-[400px]"
        >
          <CardHeader className="h-[45%] p-4">
            <CardTitle>{card.title}</CardTitle>
            <CardDescription>{card.description}</CardDescription>
          </CardHeader>
          <Separator />
          <CardContent className="flex h-[60%] items-center justify-center">
            <span className="flex items-center justify-center text-xl">
              {card.content}
            </span>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
