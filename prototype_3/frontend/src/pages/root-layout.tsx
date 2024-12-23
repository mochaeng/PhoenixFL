import { AppSidebar } from "@/components/sidebar/app-sidebar";
import { Separator } from "@/components/ui/separator";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Outlet } from "react-router-dom";
import { HeaderBreadcrumb } from "@/components/header-breadcrumb";
import { WebsocketListener } from "@/components/websocket-listener";
import React from "react";
import { WebsocketStatus } from "@/components/websocket-status";

const Header = React.memo(function Header() {
  return (
    <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-[[data-collapsible=icon]]/sidebar-wrapper:h-12">
      <div className="flex items-center gap-2">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mr-2 h-4" />
        <HeaderBreadcrumb />
      </div>
      <WebsocketStatus className="flex flex-1 justify-end text-sm" />
    </header>
  );
});

function RootLayout() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="px-4">
        {/* <PacketStoreProvider> */}
        <Header />
        <WebsocketListener />
        <div className="flex justify-center bg-rose-50">
          <Outlet />
        </div>
        {/* </PacketStoreProvider> */}
      </SidebarInset>
    </SidebarProvider>
  );
}

export default RootLayout;
