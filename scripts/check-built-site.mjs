import { access, readFile } from "node:fs/promises";
import { extname } from "node:path";

const root = new URL("../dist/", import.meta.url);
const pages = [
  "index.html",
  "results.html",
  "methodology.html",
  "technical.html",
  "api.html",
  "lightgcn-results.html",
  "lightgcn-methodology.html",
];

async function exists(path) {
  try {
    await access(new URL(path, root));
    return true;
  } catch {
    return false;
  }
}

for (const required of [...pages, "reports/lightgcn.json", "og.png", "server/index.js"]) {
  if (!(await exists(required))) throw new Error(`Missing built site asset: ${required}`);
}

for (const page of pages) {
  const html = await readFile(new URL(page, root), "utf8");
  for (const marker of ["TODO", "TBD", "REPLACE_ME"]) {
    if (html.includes(marker)) throw new Error(`${page} contains ${marker}`);
  }
  for (const requiredMeta of ["og:title", "og:description", "og:image", "twitter:card"]) {
    if (!html.includes(requiredMeta)) throw new Error(`${page} is missing ${requiredMeta}`);
  }
  const image = html.match(/<meta property="og:image" content="([^"]+)"/i)?.[1];
  if (!image || !image.startsWith("https://")) {
    throw new Error(`${page} must use a trusted absolute HTTPS social image URL`);
  }
  const links = [...html.matchAll(/(?:href|src)="([^"]+)"/g)].map((match) => match[1]);
  for (const rawLink of links) {
    if (/^(?:https?:|mailto:|tel:|#|data:)/.test(rawLink)) continue;
    const clean = rawLink.split(/[?#]/, 1)[0].replace(/^\//, "");
    if (!clean || (!extname(clean) && !clean.endsWith("/"))) continue;
    if (!(await exists(clean))) {
      throw new Error(`${page} links to missing built asset: ${rawLink}`);
    }
  }
}

const report = JSON.parse(await readFile(new URL("reports/lightgcn.json", root), "utf8"));
if (report.status !== "verified") throw new Error("Built LightGCN report is not verified");
if (!Number.isFinite(report.test_metrics?.["recall@20"])) {
  throw new Error("Built LightGCN report lacks a finite Recall@20");
}
