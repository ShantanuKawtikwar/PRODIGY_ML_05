const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const clearBtn = document.getElementById('clear-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const btnText = document.querySelector('.btn-text');
const loader = document.querySelector('.loader');

const resultsSection = document.getElementById('results-section');
const foodClassEl = document.getElementById('food-class');
const foodCaloriesEl = document.getElementById('food-calories');
const foodConfidenceEl = document.getElementById('food-confidence');
const errorMessageEl = document.getElementById('error-message');

let currentFile = null;

// Event Listeners for UI interaction
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

clearBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // prevent clicking the dropzone beneath
    clearPreview();
});

analyzeBtn.addEventListener('click', analyzeImage);

// Logic
function handleFile(file) {
    if (!file.type.match('image.*')) {
        showError("Please upload an image file (JPEG, PNG, WEBP).");
        return;
    }

    currentFile = file;
    const reader = new FileReader();

    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;
        hideError();
        resultsSection.classList.add('hidden'); // hide past results
    };

    reader.readAsDataURL(file);
}

function clearPreview() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    dropZone.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    analyzeBtn.disabled = true;
    resultsSection.classList.add('hidden');
    hideError();
}

async function analyzeImage() {
    if (!currentFile) return;

    setLoadingState(true);
    hideError();
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('image', currentFile);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        displayResults(data);
    } catch (error) {
        console.error("Analysis failed:", error);
        showError(error.message || "Failed to analyze image. Please try again.");
    } finally {
        setLoadingState(false);
    }
}

function displayResults(data) {
    // Formatting food name (e.g., apple_pie -> Apple Pie)
    const formattedName = data.food.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');

    foodClassEl.textContent = formattedName;
    
    if (typeof data.calories === 'number') {
        foodCaloriesEl.textContent = `~${data.calories} kcal`;
    } else {
        foodCaloriesEl.textContent = data.calories || 'Unknown';
    }
    
    foodConfidenceEl.textContent = `${data.confidence.toFixed(2)}%`;
    
    resultsSection.classList.remove('hidden');
    
    // Smooth scroll if needed
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setLoadingState(isLoading) {
    analyzeBtn.disabled = isLoading;
    if (isLoading) {
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
    } else {
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
    }
}

function showError(message) {
    errorMessageEl.textContent = message;
    errorMessageEl.classList.remove('hidden');
}

function hideError() {
    errorMessageEl.classList.add('hidden');
}
