import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatSeconds(value: number) {
  const formatted = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
  });
  return `${formatted.format(value)}s`;
}

export function formatMaliciousPercentage(value: number) {
  const formatted = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
  });
  return `${formatted.format(value)}`;
}

export function formatDuration(seconds: number) {
  if (seconds < 1) {
    const milliseconds = Math.round(seconds * 1000);
    return `${milliseconds}ms`;
  } else if (seconds < 60) {
    return `${Math.round(seconds)} second${seconds !== 1 ? "s" : ""}`;
  } else if (seconds < 3600) {
    const minutes = Math.round(seconds / 60);
    return `${minutes} minute${minutes !== 1 ? "s" : ""}`;
  } else {
    const hours = Math.round(seconds / 3600);
    return `${hours} hour${hours !== 1 ? "s" : ""}`;
  }
}
