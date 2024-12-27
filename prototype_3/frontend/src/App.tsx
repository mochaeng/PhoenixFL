import "./App.css";
import {
  createBrowserRouter,
  Navigate,
  RouterProvider,
} from "react-router-dom";
import ErrorPage from "./pages/error-page";
import RootLayout from "./pages/root-layout";
import { OverviewPage } from "./pages/overview-page";

const router = createBrowserRouter([
  {
    path: "/",
    element: <RootLayout />,
    errorElement: <ErrorPage />,
    children: [
      { index: true, element: <Navigate to="/dashboard/overview" replace /> },
      {
        path: "dashboard",
        children: [
          { index: true, element: <Navigate to="overview" replace /> },
          { path: "overview", element: <OverviewPage /> },
          { path: "recent", element: <div>Recent</div> },
          { path: "malicious", element: <div>Malicious activity</div> },
          { path: "performance", element: <div>Performance</div> },
        ],
      },
      {
        path: "historical",
        children: [
          { index: true, element: <Navigate to="search" replace /> },
          { path: "search", element: <div>Search</div> },
        ],
      },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;
