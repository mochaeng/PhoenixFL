import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from "@/components/ui/sidebar";
import {
  AudioWaveform,
  Command,
  GalleryVerticalEnd,
  Search,
  Settings2,
  Share,
  SquareTerminal,
} from "lucide-react";
import { NavMain } from "./nav-main";
import { TeamSwitcher } from "./team-switcher";
import { NavUser } from "./nav-user";
import { NavHistory } from "./nav-history";

const data = {
  user: {
    name: "Amelia Gray",
    email: "amelia@vogue.com",
    avatar:
      "https://i.pinimg.com/736x/4e/18/76/4e187641fd96baa75eac4baf3cdb7d7b.jpg",
  },
  teams: [
    {
      name: "Phoenix FL",
      logo: GalleryVerticalEnd,
      plan: "Free",
    },
    {
      name: "Fhoenix FL",
      logo: AudioWaveform,
      plan: "Basic",
    },
    {
      name: "Phoenix FL",
      logo: Command,
      plan: "Enterprise",
    },
  ],
  navMain: [
    {
      title: "Home",
      url: "#",
      icon: SquareTerminal,
      isActive: true,
      items: [
        {
          title: "Overview",
          url: "#",
        },
        {
          title: "Most recent",
          url: "#",
        },
        {
          title: "Malicious activity",
          url: "#",
        },
        {
          title: "System performance",
          url: "#",
        },
      ],
    },
    {
      title: "Settings",
      url: "#",
      icon: Settings2,
      items: [
        {
          title: "Theme",
          url: "#",
        },
      ],
    },
  ],
  historical: [
    {
      name: "Search",
      url: "#",
      icon: Search,
    },
    {
      name: "Export",
      url: "#",
      icon: Share,
    },
  ],
};

export function AppSidebar() {
  return (
    <Sidebar collapsible="icon" variant="inset">
      <SidebarHeader>
        <TeamSwitcher teams={data.teams} />
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        <NavHistory data={data.historical} />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter>
    </Sidebar>
  );
}
