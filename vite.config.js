import { copyFile, mkdir } from "node:fs/promises";
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
  "multvae-yelp2018-results.html",
];

const sitesWorker = {
  name: "deep-recsys-sites-worker",
  apply: "build",
  async closeBundle() {
    if (process.env.SITES_CLIENT_ONLY === "1") return;
    await mkdir(resolve("dist/server"), { recursive: true });
    await copyFile(resolve("site-worker.js"), resolve("dist/server/index.js"));
  },
};

export default defineConfig({
  plugins: process.env.SITES_CLIENT_ONLY === "1" ? [] : [sites(), sitesWorker],
  build: {
    rollupOptions: {
      input: Object.fromEntries(pages.map((page) => [page.replace(".html", ""), resolve(page)])),
    },
  },
});
