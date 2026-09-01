import { readFile } from "node:fs/promises";

const reportPath = new URL("../public/reports/lightgcn.json", import.meta.url);
let report;

try {
  report = JSON.parse(await readFile(reportPath, "utf8"));
} catch (error) {
  throw new Error(`Verified LightGCN report is required for production: ${error.message}`);
}

if (report.status !== "verified") {
  throw new Error("LightGCN report must have status=verified");
}

for (const metric of ["recall@20", "ndcg@20"]) {
  if (!Number.isFinite(report.test_metrics?.[metric])) {
    throw new Error(`LightGCN report is missing finite ${metric}`);
  }
}

const serialized = JSON.stringify(report).toLowerCase();
for (const marker of ["todo", "tbd", "placeholder", "replace_me"]) {
  if (serialized.includes(marker)) {
    throw new Error(`LightGCN report contains forbidden placeholder marker: ${marker}`);
  }
}

if (!/^[a-f0-9]{64}$/.test(report.report_hash ?? "")) {
  throw new Error("LightGCN aggregate report is missing its SHA-256 provenance hash");
}
