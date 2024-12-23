import { LatestPacketsTable } from "@/components/home/latest-packets-table";
import { WebsocketStatus } from "@/components/websocket-status";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { usePacketStore } from "@/hooks/use-packet-store";

export function OverviewPage() {
  const totalCount = usePacketStore((state) => state.totalCount);

  return (
    <div className="flex flex-col gap-2">
      <WebsocketStatus />
      <Card className="w-[450px]">
        <CardHeader>
          <CardTitle>Total Classified Packets</CardTitle>
          <CardDescription>
            The amount of packets PhoenixFL has already classified
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p>{totalCount}</p>
        </CardContent>
        <CardFooter>
          <p>Card Footer</p>
        </CardFooter>
      </Card>
      {/* <LatestPacketsTable /> */}
    </div>
  );
}
