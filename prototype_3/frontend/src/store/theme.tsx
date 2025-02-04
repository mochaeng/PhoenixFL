import { create } from "zustand";

type Theme = "dark" | "light" | "system";

type ThemeStore = {
  theme: Theme;
  setTheme: (theme: Theme) => void;
};

const storageKey = "phoenix-ui-theme";
const defaultTheme: Theme = "light";

export const useThemeStore = create<ThemeStore>()((set) => {
  const initialTheme =
    (localStorage.getItem(storageKey) as Theme) || defaultTheme;

  const root = window.document.documentElement;
  root.classList.remove("light", "dark");

  if (initialTheme === "system") {
    const systemTheme = window.matchMedia("(prefers-color-scheme: dark)")
      .matches
      ? "dark"
      : "light";

    root.classList.add(systemTheme);
  } else {
    root.classList.add(initialTheme);
  }

  return {
    theme: initialTheme,
    setTheme: (theme: Theme) => {
      localStorage.setItem(storageKey, theme);
      set({ theme });
      const root = window.document.documentElement;
      root.classList.remove("light", "dark");
      if (theme === "system") {
        const systemTheme = window.matchMedia("(prefers-color-scheme: dark)")
          .matches
          ? "dark"
          : "light";
        root.classList.add(systemTheme);
      } else {
        root.classList.add(theme);
      }
    },
  };
});
