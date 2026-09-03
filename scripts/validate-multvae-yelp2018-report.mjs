import { readFile } from "node:fs/promises";

const reportPath = new URL("../public/reports/multvae-yelp2018.json", import.meta.url);
let report;

try {
  report = JSON.parse(await readFile(reportPath, "utf8"));
} catch (error) {
  throw new Error(`Verified Multi-VAE Yelp2018 report is required for production: ${error.message}`);
}

if (report.status !== "verified" || report.dataset?.key !== "yelp2018") {
  throw new Error("Multi-VAE Yelp2018 report must be verified and identify dataset=yelp2018");
}

for (const k of [10, 20, 50, 100]) {
  for (const prefix of ["recall", "ndcg"]) {
    const key = `${prefix}@${k}`;
    if (!Number.isFinite(report.test_metrics?.[key])) {
      throw new Error(`Multi-VAE Yelp2018 report is missing finite ${key}`);
    }
    if (!Number.isFinite(report.validation_metrics?.[key])) {
      throw new Error(`Multi-VAE Yelp2018 report is missing finite validation ${key}`);
    }
  }
}

if (!report.training_curve?.length || !report.final_training_curve?.length) {
  throw new Error("Multi-VAE Yelp2018 report must include training and final-retrain curves");
}

if (!/^[a-f0-9]{64}$/.test(report.report_hash ?? "")) {
  throw new Error("Multi-VAE Yelp2018 aggregate report is missing its SHA-256 provenance hash");
}

const serialized = JSON.stringify(report).toLowerCase();
for (const marker of ["todo", "tbd", "placeholder", "replace_me"]) {
  if (serialized.includes(marker)) {
    throw new Error(`Multi-VAE Yelp2018 report contains forbidden placeholder marker: ${marker}`);
  }
}

for (const forbidden of ["user_ids", "item_ids", "checkpoint_state", "raw_ratings"]) {
  if (Object.prototype.hasOwnProperty.call(report, forbidden)) {
    throw new Error(`Public report must not expose ${forbidden}`);
  }
}
