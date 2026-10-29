// ClaimTrackr - small progressive enhancements (the app also works without JavaScript)
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("claim-form");
  if (!form) return;

  // --- drag & drop PDF upload ---
  const zone = document.getElementById("dropzone");
  const input = document.getElementById("medical_bill");
  const label = document.getElementById("dz-label");
  const setFile = (file) => {
    if (!file) return;
    label.textContent = file.name + "  (" + Math.max(1, Math.round(file.size / 1024)) + " KB)";
    zone.classList.add("has-file");
  };
  input.addEventListener("change", () => setFile(input.files[0]));
  ["dragenter", "dragover"].forEach((e) => zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((e) => zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (ev) => {
    if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; setFile(input.files[0]); }
  });

  // --- "Fill form" buttons for the sample claims ---
  document.querySelectorAll("[data-fill]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const values = JSON.parse(btn.dataset.fill);
      Object.entries(values).forEach(([key, value]) => { if (form.elements[key]) form.elements[key].value = value; });
      form.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  // --- loading overlay (LLM calls can take a few seconds) ---
  form.addEventListener("submit", () => {
    if (form.checkValidity()) document.getElementById("overlay").classList.add("show");
  });
  window.addEventListener("pageshow", () => document.getElementById("overlay").classList.remove("show")); // back button
});
