import { IpsTable } from "@/components/home/ips-table";
import { LatestPacketsTable } from "@/components/home/latest-packets-table";

export function RecentPage() {
  return (
    <div className="flex h-full w-full max-w-screen-3xl flex-col items-center justify-center gap-6">
      <LatestPacketsTable />
      <IpsTable className="max-w-[950px]" />
    </div>
  );
}
