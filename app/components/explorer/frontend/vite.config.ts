import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Streamlit serves the built component from a sub-path, so assets must be relative.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1200 },
  server: { port: 5173 },
});
