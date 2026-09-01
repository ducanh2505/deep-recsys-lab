import { resolve } from "node:path";

import { sites } from "@openai/sites-vite-plugin";
import { defineConfig } from "vite";

const pages = [
  "index.html",
  "results.html",
  "methodology.html",
  "technical.html",
  "api.html",
  "lightgcn-results.html",
  "lightgcn-methodology.html",
];

export default defineConfig({
  plugins: process.env.SITES_CLIENT_ONLY === "1" ? [] : [sites()],
  build: {
    rollupOptions: {
      input: Object.fromEntries(pages.map((page) => [page.replace(".html", ""), resolve(page)])),
    },
  },
});
