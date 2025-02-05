import { Label, Pie, PieChart } from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { usePacketStore } from "@/store/packet";
import { cn } from "@/lib/utils";

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

const formatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  compactDisplay: "short",
});

export function PacketsPieChart({
  className,
}: React.HTMLAttributes<HTMLDivElement>) {
  const stats = usePacketStore((state) => state.stats);

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

  return (
    <Card className={cn("flex flex-col", className)}>
      <CardHeader className="items-center pb-0">
        <CardTitle>Malicious Packets</CardTitle>
        <CardDescription>
          Distribution of malicious and normal network packets.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[250px]"
        >
          <PieChart>
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
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
                          Packets
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
      <CardFooter className="flex-col gap-2 text-sm">
        <div className="flex items-center gap-2 font-medium leading-none">
          Proportion of malicious and normal network traffic.
        </div>
        <div className="leading-none text-muted-foreground">
          Useful for tracking malicious activity over time.
        </div>
      </CardFooter>
    </Card>
  );
}
