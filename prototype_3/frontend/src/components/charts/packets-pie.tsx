import { Label, Pie, PieChart } from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "../ui/chart";
import { usePacketStore } from "@/store/packet";
import { cn } from "@/lib/utils";

export function PacketsPieChart({
  className,
}: React.HTMLAttributes<HTMLDivElement>) {
  const stats = usePacketStore((state) => state.stats);

  const formatter = new Intl.NumberFormat("en-US", {
    notation: "compact",
    compactDisplay: "short",
  });

  const chartData = [
    {
      category: "normal",
      packets: stats.total_packets - stats.total_malicious,
      fill: "var(--color-normal)",
    },
    {
      category: "malicious",
      packets: stats.total_malicious,
      fill: "var(--color-malicious)",
    },
  ];

  const chartConfig = {
    normal: {
      label: "Normal",
      color: "hsl(var(--chart-2))",
    },
    malicious: {
      label: "Malicious",
      color: "hsl(var(--chart-1))",
    },
  } satisfies ChartConfig;

  return (
    <Card className={cn("flex w-full flex-col", className)}>
      <CardHeader className="items-center pb-0">
        <CardTitle>Malicious Packets - Pie Chart</CardTitle>
        <CardDescription>Normal and Malicious packets</CardDescription>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[250px]"
        >
          <PieChart>
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent className="" hideLabel />}
            />
            <Pie
              data={chartData}
              dataKey="packets"
              nameKey="category"
              innerRadius={60}
              strokeWidth={5}
            >
              <Label
                content={({ viewBox }) => {
                  if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                    return (
                      <text
                        x={viewBox.cx}
                        y={viewBox.cy}
                        textAnchor="middle"
                        dominantBaseline="middle"
                      >
                        <tspan
                          x={viewBox.cx}
                          y={viewBox.cy}
                          className="fill-foreground text-3xl font-bold"
                        >
                          {formatter.format(stats.total_packets)}
                        </tspan>
                        <tspan
                          x={viewBox.cx}
                          y={(viewBox.cy || 0) + 24}
                          className="fill-muted-foreground"
                        >
                          packets
                        </tspan>
                      </text>
                    );
                  }
                }}
              />
            </Pie>
          </PieChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
