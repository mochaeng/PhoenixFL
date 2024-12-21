import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import React from "react";
import { useLocation } from "react-router-dom";

export function HeaderBreadcrumb() {
  const location = useLocation();

  const segments = location.pathname
    .split("/")
    .filter((segment) => segment.length > 0);

  const pathsURI = segments.reduce<string[]>((acc, segment, idx) => {
    const path = `${acc[idx - 1] || ""}/${segment}`;
    acc.push(path);
    return acc;
  }, []);

  const formattedSegments = segments.map(
    (segment) => segment.charAt(0).toUpperCase() + segment.slice(1),
  );

  return (
    <Breadcrumb>
      <BreadcrumbList>
        {segments.map((_, idx) => (
          <React.Fragment key={idx}>
            <BreadcrumbItem className="hidden md:block">
              {idx < segments.length - 1 ? (
                <BreadcrumbLink href={pathsURI[idx]}>
                  {formattedSegments[idx]}
                </BreadcrumbLink>
              ) : (
                <BreadcrumbPage>{formattedSegments[idx]}</BreadcrumbPage>
              )}
            </BreadcrumbItem>
            {idx < segments.length - 1 ? (
              <BreadcrumbSeparator className="hidden md:block" />
            ) : null}
          </React.Fragment>
        ))}
      </BreadcrumbList>
    </Breadcrumb>
  );
}
