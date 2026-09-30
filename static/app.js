const contentInput = document.getElementById("content-file");
const styleInput = document.getElementById("style-file");
const strengthInput = document.getElementById("strength");
const message = document.getElementById("message");
const createButton = document.getElementById("create-button");
const progressPanel = document.getElementById("progress-panel");
const progressBar = document.getElementById("render-progress");
const progressMessage = document.getElementById("progress-message");
const progressPercent = document.getElementById("progress-percent");

function showPreview(input, zone) {
  const file = input.files[0];
  if (!file) return;

  const image = zone.querySelector("img");
  image.src = URL.createObjectURL(file);
  zone.classList.add("has-image");
}

contentInput.addEventListener("change", () => {
  showPreview(contentInput, document.getElementById("content-zone"));
});

styleInput.addEventListener("change", () => {
  showPreview(styleInput, document.getElementById("style-zone"));
});

strengthInput.addEventListener("input", () => {
  document.getElementById("strength-label").textContent =
    `${strengthInput.value}%`;
});

let progressRequestRunning = false;

async function refreshProgress() {
  if (progressRequestRunning) return;
  progressRequestRunning = true;

  try {
    const response = await fetch("/api/progress", { cache: "no-store" });
    if (!response.ok) return;

    const data = await response.json();
    progressBar.value = data.progress || 0;
    progressPercent.textContent = `${data.progress || 0}%`;
    if (data.message) progressMessage.textContent = data.message;
  } catch (_error) {
    // The main request displays the user-facing error if the server is down.
  } finally {
    progressRequestRunning = false;
  }
}

createButton.addEventListener("click", async () => {
  if (!contentInput.files[0] || !styleInput.files[0]) {
    message.textContent = "Please choose both images first.";
    return;
  }

  createButton.disabled = true;
  progressPanel.hidden = false;
  progressBar.value = 0;
  progressPercent.textContent = "0%";
  progressMessage.textContent = "Uploading and checking your images…";
  document.getElementById("button-label").textContent =
    "Creating your artwork…";
  message.textContent = "Your image stays on this computer while it is processed.";

  const form = new FormData();
  form.append("content", contentInput.files[0]);
  form.append("style", styleInput.files[0]);
  form.append("strength", Number(strengthInput.value) / 100);

  const progressTimer = window.setInterval(refreshProgress, 700);

  try {
    await refreshProgress();
    const response = await fetch("/api/stylize", {
      method: "POST",
      body: form,
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The request failed.");

    const imageUrl = `${data.image_url}?time=${Date.now()}`;
    document.getElementById("result-image").src = imageUrl;
    document.getElementById("download-link").href = imageUrl;

    const resultSection = document.getElementById("result-section");
    resultSection.classList.add("visible");
    message.textContent = "Your artwork is ready.";
    resultSection.scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    message.textContent = error.message;
  } finally {
    window.clearInterval(progressTimer);
    await refreshProgress();
    createButton.disabled = false;
    document.getElementById("button-label").textContent =
      "Create my artwork";
  }
});
