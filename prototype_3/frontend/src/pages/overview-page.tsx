import { PacketsPieChart } from "@/components/charts/packets-pie";
import { IpsTable } from "@/components/home/ips-table";
import { LatestPacketsTable } from "@/components/home/latest-packets-table";
import { MainStatsCards } from "@/components/home/main-stats-cards";

export function OverviewPage() {
  return (
    <div className="flex h-full w-full max-w-screen-3xl flex-col items-center justify-center gap-2">
      {/* <MainStatsCards />
      <LatestPacketsTable />
      <IpsTable /> */}
      <PacketsPieChart />
    </div>
  );
}
