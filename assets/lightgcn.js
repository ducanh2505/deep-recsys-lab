const languageButtons = document.querySelectorAll("[data-language]");
const translatable = document.querySelectorAll("[data-en][data-vi]");

function setLanguage(language) {
  document.documentElement.lang = language;
  translatable.forEach((element) => {
    element.textContent = element.dataset[language];
  });
  languageButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.language === language));
  });
  try {
    localStorage.setItem("deep-recsys-language", language);
  } catch (_error) {
    // Language still works when browser storage is unavailable.
  }
  document.dispatchEvent(new CustomEvent("languagechange", { detail: language }));
}

languageButtons.forEach((button) => {
  button.addEventListener("click", () => setLanguage(button.dataset.language));
});

let savedLanguage = "en";
try {
  savedLanguage = localStorage.getItem("deep-recsys-language") === "vi" ? "vi" : "en";
} catch (_error) {
  savedLanguage = "en";
}
setLanguage(savedLanguage);

const numberFormat = new Intl.NumberFormat("en-US");

async function loadReport() {
  try {
    const response = await fetch("/reports/lightgcn.json", { cache: "no-store" });
    if (!response.ok) throw new Error("report unavailable");
    const report = await response.json();
    if (report.status !== "verified") throw new Error("report is not verified");
    document.querySelectorAll("[data-report-state]").forEach((element) => {
      element.dataset.reportState = "verified";
    });
    document.querySelectorAll("[data-field]").forEach((element) => {
      const value = element.dataset.field.split(".").reduce((item, key) => item?.[key], report);
      if (typeof value === "number") {
        element.textContent =
          element.dataset.format === "metric" ? value.toFixed(4) : numberFormat.format(value);
      } else if (value !== undefined) {
        element.textContent = String(value);
      }
    });
    window.lightgcnReport = report;
    document.dispatchEvent(new CustomEvent("lightgcn-report", { detail: report }));
  } catch (_error) {
    document.documentElement.dataset.report = "pending";
  }
}

loadReport();
