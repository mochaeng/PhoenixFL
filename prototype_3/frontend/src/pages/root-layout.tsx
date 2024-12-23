import { AppSidebar } from "@/components/sidebar/app-sidebar";
import { Separator } from "@/components/ui/separator";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Outlet } from "react-router-dom";
import { HeaderBreadcrumb } from "@/components/header-breadcrumb";
import { PacketStoreProvider } from "@/store/packet-provider";
import { WebsocketListener } from "@/components/websocket-listener";
import React from "react";

const Header = React.memo(function Header() {
  return (
    <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-[[data-collapsible=icon]]/sidebar-wrapper:h-12">
      <div className="flex items-center gap-2">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mr-2 h-4" />
        <HeaderBreadcrumb />
      </div>
    </header>
  );
});

function RootLayout() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="px-4">
        <Header />
        <PacketStoreProvider>
          <WebsocketListener />
          <div className="">
            <Outlet />
          </div>
        </PacketStoreProvider>
      </SidebarInset>
    </SidebarProvider>
  );
}

export default RootLayout;
