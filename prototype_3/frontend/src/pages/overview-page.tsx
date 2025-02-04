import { IpsBarChart } from "@/components/charts/ips-bar";
import { PacketsPieChart } from "@/components/charts/packets-pie";
import { WorkersBarChart } from "@/components/charts/workers-bar";
import { IpsTable } from "@/components/home/ips-table";
import { LatestPacketsTable } from "@/components/home/latest-packets-table";
import { MainStatsCards } from "@/components/home/main-stats-cards";

export function OverviewPage() {
  return (
    <div className="flex h-full w-full max-w-screen-3xl flex-col items-center justify-center gap-6">
      <MainStatsCards />
      {/* <LatestPacketsTable /> */}
      {/* <IpsTable /> */}
      <div className="flex w-full flex-wrap justify-center bg-yellow-500">
        <div className="inline-flex w-full flex-wrap justify-center gap-4">
          <PacketsPieChart className="w-[450px]" />
          <WorkersBarChart className="w-[450px]" />
        </div>
      </div>
    </div>
  );
}
