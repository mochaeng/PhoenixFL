import { MainStatsCards } from "@/components/home/main-stats-cards";

export function OverviewPage() {
  return (
    <div className="max-w-screen-3xl flex h-full w-full flex-col justify-center gap-2 bg-blue-100">
      <MainStatsCards />
      {/* <LatestPacketsTable /> */}
    </div>
  );
}
