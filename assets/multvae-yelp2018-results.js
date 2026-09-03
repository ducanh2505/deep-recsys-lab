const numberFormat = new Intl.NumberFormat("en-US");
const metricKeys = [10, 20, 50, 100];

function finite(value) {
  return Number.isFinite(Number(value));
}

function formatMetric(value) {
  return finite(value) ? Number(value).toFixed(4) : "—";
}

function formatScalar(value, digits = 4) {
  return finite(value) ? Number(value).toFixed(digits) : "—";
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secondsPart = total % 60;
  return (
    String(hours) +
    "h " +
    String(minutes).padStart(2, "0") +
    "m " +
    String(secondsPart).padStart(2, "0") +
    "s"
  );
}

function setText(selector, value) {
  const element = document.querySelector(selector);
  if (element) element.textContent = String(value);
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(canvas.clientWidth, 320);
  const height = Math.max(canvas.clientHeight, 220);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function drawLineChart(canvasId, records, series, xKey = "epoch") {
  const canvas = document.querySelector("#" + canvasId);
  if (!canvas) return;
  const valid = records.filter((record) =>
    series.some(({ value }) => finite(value(record))),
  );
  if (!valid.length) return;
  const { context, width, height } = resizeCanvas(canvas);
  const margin = { top: 18, right: 18, bottom: 36, left: 54 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const xValues = valid.map((record) => Number(record[xKey]));
  const values = valid.flatMap((record) =>
    series
      .map(({ value }) => Number(value(record)))
      .filter((item) => Number.isFinite(item)),
  );
  let low = Math.min(...values);
  let high = Math.max(...values);
  if (low === high) {
    low -= 1;
    high += 1;
  } else {
    const padding = (high - low) * 0.08;
    low = Math.max(0, low - padding);
    high += padding;
  }
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const xScale = (value) =>
    margin.left + ((value - minX) / Math.max(1, maxX - minX)) * innerWidth;
  const yScale = (value) =>
    margin.top + ((high - value) / Math.max(1e-12, high - low)) * innerHeight;

  context.font = "11px SFMono-Regular, monospace";
  context.fillStyle = "#7b8798";
  context.strokeStyle = "#e1e6ee";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = margin.top + (innerHeight * tick) / 4;
    const value = high - ((high - low) * tick) / 4;
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(width - margin.right, y);
    context.stroke();
    context.fillText(value.toFixed(3), 4, y + 4);
  }
  context.fillText(xKey, width - margin.right - 42, height - 8);

  series.forEach(({ color, value }) => {
    const points = valid
      .map((record) => ({ record, y: Number(value(record)) }))
      .filter((point) => Number.isFinite(point.y));
    if (!points.length) return;
    context.beginPath();
    points.forEach(({ record, y }, index) => {
      const x = xScale(Number(record[xKey]));
      const pointY = yScale(y);
      if (index === 0) context.moveTo(x, pointY);
      else context.lineTo(x, pointY);
    });
    context.strokeStyle = color;
    context.lineWidth = 2.5;
    context.stroke();
  });
}

function drawTestChart(report) {
  const canvas = document.querySelector("#test-chart");
  if (!canvas) return;
  const points = metricKeys
    .map((k) => ({
      k,
      recall: Number(report.test_metrics?.["recall@" + k]),
      ndcg: Number(report.test_metrics?.["ndcg@" + k]),
    }))
    .filter((point) => Number.isFinite(point.recall) || Number.isFinite(point.ndcg));
  if (!points.length) return;
  const { context, width, height } = resizeCanvas(canvas);
  const margin = { top: 18, right: 18, bottom: 42, left: 48 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const high =
    Math.max(
      0.1,
      ...points.flatMap((point) => [point.recall, point.ndcg].filter(Number.isFinite)),
    ) * 1.12;
  const yScale = (value) => margin.top + (1 - value / high) * innerHeight;
  context.font = "11px SFMono-Regular, monospace";
  context.fillStyle = "#7b8798";
  context.strokeStyle = "#e1e6ee";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = margin.top + (innerHeight * tick) / 4;
    const value = high - (high * tick) / 4;
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(width - margin.right, y);
    context.stroke();
    context.fillText(value.toFixed(3), 4, y + 4);
  }
  const groupWidth = innerWidth / points.length;
  const barWidth = Math.min(28, groupWidth * 0.24);
  points.forEach((point, index) => {
    const center = margin.left + groupWidth * (index + 0.5);
    [
      { value: point.recall, color: "#315ee8", offset: -barWidth * 0.62 },
      { value: point.ndcg, color: "#25b899", offset: barWidth * 0.62 },
    ].forEach(({ value, color, offset }) => {
      if (!Number.isFinite(value)) return;
      const x = center + offset - barWidth / 2;
      const y = yScale(value);
      context.fillStyle = color;
      context.fillRect(x, y, barWidth, margin.top + innerHeight - y);
    });
    context.fillStyle = "#7b8798";
    context.fillText("@" + point.k, center - 10, height - 12);
  });
}

function renderTestTable(report) {
  const body = document.querySelector("#test-table tbody");
  if (!body) return;
  body.replaceChildren();
  metricKeys.forEach((k) => {
    const row = document.createElement("tr");
    [
      k,
      formatMetric(report.test_metrics?.["recall@" + k]),
      formatMetric(report.test_metrics?.["ndcg@" + k]),
    ].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    });
    body.append(row);
  });
}

function renderSweepTable(report) {
  const body = document.querySelector("#sweep-table tbody");
  if (!body) return;
  body.replaceChildren();
  const winnerId = report.selection?.winner?.id;
  ["round1", "round2", "full"].forEach((stage) => {
    (report.sweep?.[stage] || []).forEach((record) => {
      const metrics = record.best_validation_metrics || record.validation_metrics || {};
      const row = document.createElement("tr");
      if (record.trial?.id === winnerId && stage === "full") row.className = "winner-row";
      [
        stage,
        record.trial?.id || "—",
        record.best_epoch ?? record.completed_epoch ?? "—",
        formatMetric(metrics["recall@20"]),
        formatMetric(metrics["ndcg@20"]),
        formatMetric(metrics["recall@100"]),
        formatMetric(metrics["ndcg@100"]),
      ].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.append(cell);
      });
      body.append(row);
    });
  });
}

function renderCurveTable(report) {
  const body = document.querySelector("#curve-table tbody");
  if (!body) return;
  body.replaceChildren();
  const records = report.training_curve || [];
  const visible = records.length > 60 ? records.slice(-60) : records;
  visible.forEach((record) => {
    const train = record.train || {};
    const validation = record.validation || {};
    const row = document.createElement("tr");
    [
      record.epoch ?? "—",
      formatScalar(train.loss),
      formatScalar(train.nll),
      formatScalar(train.kl),
      formatScalar(train.beta),
      formatMetric(validation["recall@20"]),
      formatMetric(validation["ndcg@20"]),
    ].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    });
    body.append(row);
  });
}

function renderWinner(report) {
  const winner = report.selection?.winner || {};
  setText("#winner-architecture", (winner.hidden_dims || []).join(" → ") || "—");
  setText("#winner-dropout", formatScalar(winner.dropout));
  setText("#winner-beta", formatScalar(winner.beta_cap));
  setText("#winner-lr", winner.learning_rate ?? "—");
  setText("#winner-wd", winner.weight_decay ?? "—");
  setText("#best-epoch", report.selection?.best_epoch ?? "—");
}

function renderReport(report) {
  document.querySelectorAll(".metric-card").forEach((card) => {
    card.dataset.state = "verified";
  });
  document.querySelectorAll("[data-report-state]").forEach((element) => {
    element.dataset.reportState = "verified";
  });
  const statusTag = document.querySelector(".status-tag");
  const statusCopy = document.querySelector(".status-copy");
  if (statusTag && statusCopy) {
    statusTag.dataset.en = "VERIFIED RUN";
    statusTag.dataset.vi = "RUN ĐÃ XÁC MINH";
    statusCopy.dataset.en =
      "Tuning, clean retrain and the one-time test evaluation completed with matching hashes.";
    statusCopy.dataset.vi =
      "Tuning, retrain sạch và lần đánh giá test duy nhất đã hoàn tất với checksum khớp.";
    const language = document.documentElement.lang === "vi" ? "vi" : "en";
    statusTag.textContent = statusTag.dataset[language];
    statusCopy.textContent = statusCopy.dataset[language];
  }
  metricKeys.forEach((k) => {
    setText("#recall-" + k, formatMetric(report.test_metrics?.["recall@" + k]));
  });
  [20, 100].forEach((k) => {
    setText("#ndcg-" + k, formatMetric(report.test_metrics?.["ndcg@" + k]));
  });
  renderTestTable(report);
  renderSweepTable(report);
  renderCurveTable(report);
  renderWinner(report);
  setText("#runtime-value", formatDuration(report.duration_seconds));
  const environment = report.environment || {};
  setText(
    "#hardware-value",
    (environment.device || "—") +
      " · " +
      (environment.precision || "fp32") +
      " · " +
      (environment.torch_threads || "—") +
      " threads",
  );
  setText("#dataset-hash", report.provenance?.dataset_hash || "—");
  setText(
    "#catalog-value",
    numberFormat.format(report.dataset?.counts?.users || 0) +
      " × " +
      numberFormat.format(report.dataset?.counts?.items || 0),
  );
  setText("#report-hash", report.report_hash || "—");
  drawLineChart(
    "loss-chart",
    report.training_curve || [],
    [
      { color: "#315ee8", value: (record) => record.train?.loss },
      { color: "#e8a93e", value: (record) => record.train?.nll },
      { color: "#e66e58", value: (record) => record.train?.kl },
    ],
  );
  drawLineChart(
    "validation-chart",
    (report.training_curve || []).filter((record) =>
      finite(record.validation?.["recall@20"]),
    ),
    [
      { color: "#315ee8", value: (record) => record.validation?.["recall@20"] },
      { color: "#25b899", value: (record) => record.validation?.["ndcg@20"] },
    ],
  );
  drawTestChart(report);
  window.multvaeYelp2018Report = report;
  document.dispatchEvent(new CustomEvent("multvae-yelp2018-report", { detail: report }));
}

document.addEventListener("languagechange", () => {
  const report = window.multvaeYelp2018Report;
  if (!report) return;
  const statusTag = document.querySelector(".status-tag");
  const statusCopy = document.querySelector(".status-copy");
  const language = document.documentElement.lang === "vi" ? "vi" : "en";
  if (statusTag?.dataset[language]) statusTag.textContent = statusTag.dataset[language];
  if (statusCopy?.dataset[language]) statusCopy.textContent = statusCopy.dataset[language];
});

window.addEventListener("resize", () => {
  const report = window.multvaeYelp2018Report;
  if (report) renderReport(report);
});

async function loadReport() {
  try {
    const response = await fetch("/reports/multvae-yelp2018.json", { cache: "no-store" });
    if (!response.ok) throw new Error("report unavailable");
    const report = await response.json();
    if (report.status !== "verified") throw new Error("report is not verified");
    renderReport(report);
  } catch (_error) {
    document.documentElement.dataset.report = "pending";
  }
}

loadReport();
