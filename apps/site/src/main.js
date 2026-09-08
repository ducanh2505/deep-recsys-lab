import "./styles.css";

const navToggle = document.querySelector(".nav-toggle");
const navigation = document.querySelector(".site-nav");

navToggle?.addEventListener("click", () => {
  const isOpen = navToggle.getAttribute("aria-expanded") === "true";
  navToggle.setAttribute("aria-expanded", String(!isOpen));
  navigation?.classList.toggle("is-open", !isOpen);
});

navigation?.addEventListener("click", (event) => {
  if (event.target instanceof HTMLAnchorElement) {
    navToggle?.setAttribute("aria-expanded", "false");
    navigation.classList.remove("is-open");
  }
});

const copyButton = document.querySelector("[data-copy-target]");
copyButton?.addEventListener("click", async () => {
  const targetId = copyButton.getAttribute("data-copy-target");
  const target = targetId ? document.getElementById(targetId) : null;
  if (!target) return;

  await navigator.clipboard.writeText(target.textContent?.trim() ?? "");
  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = "Copy";
  }, 1600);
});

document.querySelectorAll("[data-year]").forEach((node) => {
  node.textContent = String(new Date().getFullYear());
});
