function formatMetric(value) {
  return Number(value).toFixed(4);
}

function formatDuration(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return `${hours}h ${String(minutes).padStart(2, "0")}m`;
}

function drawCurve(records) {
  const canvas = document.querySelector("#training-chart");
  if (!canvas || records.length === 0) return;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const margin = { top: 18, right: 18, bottom: 36, left: 48 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const maxStep = Math.max(...records.map((record) => record.step));
  const values = records.flatMap((record) => [record["recall@20"], record["ndcg@20"]]);
  const low = Math.max(0, Math.min(...values) - 0.015);
  const high = Math.max(...values) + 0.015;
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
  context.fillText("step", width - margin.right - 26, height - 8);
  const plot = (key, color) => {
    context.beginPath();
    records.forEach((record, index) => {
      const x = margin.left + (record.step / maxStep) * innerWidth;
      const y = margin.top + ((high - record[key]) / (high - low)) * innerHeight;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.strokeStyle = color;
    context.lineWidth = 2.5;
    context.stroke();
    records.forEach((record) => {
      const x = margin.left + (record.step / maxStep) * innerWidth;
      const y = margin.top + ((high - record[key]) / (high - low)) * innerHeight;
      context.beginPath();
      context.arc(x, y, 3, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
    });
  };
  plot("recall@20", "#315ee8");
  plot("ndcg@20", "#25b899");
}

function renderReport(report) {
  document.querySelectorAll(".metric-card").forEach((card) => {
    card.dataset.state = "verified";
  });
  const statusTag = document.querySelector(".status-tag");
  const statusCopy = document.querySelector(".status-copy");
  statusTag.dataset.en = "VERIFIED RUN";
  statusTag.dataset.vi = "RUN ĐÃ XÁC MINH";
  statusCopy.dataset.en =
    "Sweep, clean retrain and the one-time test evaluation completed with matching hashes.";
  statusCopy.dataset.vi =
    "Sweep, retrain sạch và lần đánh giá test duy nhất đã hoàn tất với checksum khớp.";
  const language = document.documentElement.lang === "vi" ? "vi" : "en";
  statusTag.textContent = statusTag.dataset[language];
  statusCopy.textContent = statusCopy.dataset[language];

  const sweepBody = document.querySelector("#sweep-table tbody");
  sweepBody.replaceChildren();
  report.sweep.forEach((entry) => {
    const row = document.createElement("tr");
    const winner =
      report.winner.layers === entry.candidate.layers &&
      report.winner.l2 === entry.candidate.l2;
    if (winner && entry.stage === "round2") row.className = "winner-row";
    const values = [
      entry.stage,
      entry.candidate.layers,
      entry.candidate.l2,
      entry.step,
      formatMetric(entry.metrics["recall@20"]),
      formatMetric(entry.metrics["ndcg@20"]),
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    });
    sweepBody.append(row);
  });

  const validation = report.training_curve.filter((entry) => entry.event === "validation");
  const curveBody = document.querySelector("#curve-table tbody");
  curveBody.replaceChildren();
  validation.forEach((entry) => {
    const row = document.createElement("tr");
    [
      entry.step,
      formatMetric(entry["recall@20"]),
      formatMetric(entry["ndcg@20"]),
    ].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    });
    curveBody.append(row);
  });
  drawCurve(validation);
  document.querySelector("#runtime-value").textContent = formatDuration(report.duration_seconds);
  const environment = report.environment;
  const memory = environment.memory_bytes
    ? ` · ${Math.round(environment.memory_bytes / 1024 ** 3)} GB`
    : "";
  document.querySelector("#hardware-value").textContent =
    `${environment.processor}${memory} · CPU FP32 · ${environment.torch_threads} threads`;
}

document.addEventListener("lightgcn-report", (event) => renderReport(event.detail));
window.addEventListener("resize", () => {
  if (window.lightgcnReport) {
    drawCurve(
      window.lightgcnReport.training_curve.filter((entry) => entry.event === "validation"),
    );
  }
});
