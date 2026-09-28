document.addEventListener("DOMContentLoaded", function () {
  var fileInput = document.getElementById("image");
  var preview = document.getElementById("image-preview");

  if (fileInput && preview) {
    fileInput.addEventListener("change", function () {
      var file = fileInput.files && fileInput.files[0];
      if (!file) {
        preview.style.display = "none";
        return;
      }
      var reader = new FileReader();
      reader.onload = function (e) {
        preview.src = e.target.result;
        preview.style.display = "block";
      };
      reader.readAsDataURL(file);
    });
  }

  // Disable submit button after click to avoid duplicate analysis requests
  var analysisForm = document.getElementById("scan-form");
  if (analysisForm) {
    analysisForm.addEventListener("submit", function (event) {
      var file = fileInput && fileInput.files && fileInput.files[0];

      // Keep the scanning state for submissions that can actually reach the server.
      if (!file || !analysisForm.checkValidity()) {
        event.preventDefault();
        analysisForm.reportValidity();
        return;
      }

      var submitBtn = analysisForm.querySelector('button[type="submit"]');
      var scanningPanel = document.getElementById("scanning-panel");
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Scanning...";
      }
      if (scanningPanel) {
        scanningPanel.hidden = false;
        analysisForm.hidden = true;
      }
    });
  }
});
