import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, /api is proxied to the FastAPI server so no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": process.env.VITE_BACKEND_URL || "http://localhost:8000" },
  },
  test: { environment: "node" },
});
