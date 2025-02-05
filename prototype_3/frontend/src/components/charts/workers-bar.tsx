import { Bar, BarChart, XAxis, YAxis } from "recharts";

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
import { cn } from "@/lib/utils";
import { usePacketStore } from "@/store/packet";

const chartConfig = {
  count: {
    label: "Visitors",
  },
  worker1: {
    label: "DU-1",
    color: "hsl(var(--chart-1))",
  },
  worker2: {
    label: "DU-2",
    color: "hsl(var(--chart-2))",
  },
  worker3: {
    label: "DU-3",
    color: "hsl(var(--chart-3))",
  },
  worker4: {
    label: "DU-4",
    color: "hsl(var(--chart-4))",
  },
  worker5: {
    label: "DU-5",
    color: "hsl(var(--chart-5))",
  },
} satisfies ChartConfig;

export function WorkersBarChart({
  className,
}: React.HTMLAttributes<HTMLDivElement>) {
  const chartData = [
    { key: "worker1", value: 0, fill: "var(--color-worker1)" },
    { key: "worker2", value: 0, fill: "var(--color-worker2)" },
    { key: "worker3", value: 0, fill: "var(--color-worker3)" },
    { key: "worker4", value: 0, fill: "var(--color-worker4)" },
    { key: "worker5", value: 0, fill: "var(--color-worker5)" },
  ];

  const classifications = usePacketStore(
    (state) => state.stats.workers_classifications,
  );

  classifications.forEach((classfication) => {
    const idx = chartData.findIndex((data) => data.key === classfication.key);
    if (idx > -1) {
      chartData[idx].value = classfication.value;
    }
  });

  return (
    <Card className={cn("flex w-full flex-col", className)}>
      <CardHeader>
        <CardTitle>Classifications made by DUs</CardTitle>
        <CardDescription>
          Includes both normal and malicious traffic classifications.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <BarChart
            accessibilityLayer
            data={chartData}
            layout="vertical"
            margin={{
              left: 0,
            }}
          >
            <YAxis
              dataKey="key"
              type="category"
              tickLine={false}
              tickMargin={0}
              axisLine={false}
              tickFormatter={(value) =>
                chartConfig[value as keyof typeof chartConfig]?.label
              }
            />
            <XAxis dataKey="value" type="number" hide />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Bar dataKey="value" layout="vertical" radius={5} />
          </BarChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        <div className="flex gap-2 font-medium leading-none">
          Amount of classifications made by each Detection Unit.
        </div>
        <div className="leading-none text-muted-foreground">
          Helps monitor the workload and efficiency of each Detection Unit in
          real time.
        </div>
      </CardFooter>
    </Card>
  );
}
