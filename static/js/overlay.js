// overlay.js

// Function to display the uploaded image and provide a preview before submission
function previewImage(input) {
    const file = input.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = document.getElementById("uploaded-image-preview");
            img.src = e.target.result;
            img.style.display = "block"; // Display the preview image
        };
        reader.readAsDataURL(file);
    }
}

// Optional: Basic overlay customization (if needed for positioning)
function applyOverlayPosition(xOffset, yOffset, scale) {
    const img = document.getElementById("uploaded-image-preview");
    const overlay = document.getElementById("garment-overlay");

    // Adjust position
    overlay.style.left = `${xOffset}px`;
    overlay.style.top = `${yOffset}px`;

    // Adjust scale
    overlay.style.transform = `scale(${scale})`;
}

// Handling the form submission with AJAX
document.getElementById("uploadForm").onsubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);

    // Send POST request to upload the file
    const response = await fetch("/upload", {
        method: "POST",
        body: formData,
    });

    if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        // Display the processed image
        const img = document.getElementById("processed-image");
        img.src = url;
        img.style.display = "block";
    } else {
        alert("Failed to process image.");
    }
};

// Event listener to adjust overlay when values change (optional)
document.getElementById("x-offset").addEventListener("input", (e) => {
    applyOverlayPosition(e.target.value, document.getElementById("y-offset").value, document.getElementById("scale").value);
});

document.getElementById("y-offset").addEventListener("input", (e) => {
    applyOverlayPosition(document.getElementById("x-offset").value, e.target.value, document.getElementById("scale").value);
});

document.getElementById("scale").addEventListener("input", (e) => {
    applyOverlayPosition(document.getElementById("x-offset").value, document.getElementById("y-offset").value, e.target.value);
});
