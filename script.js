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
const predictionDetailEl = document.getElementById('prediction-detail');
const errorMessageEl = document.getElementById('error-message');
const statusBanner = document.getElementById('status-banner');

let currentFile = null;
let classifier = null;
let calorieMap = {};
let modelReady = false;

const FOOD_LABEL_MAP = [
    { food: 'pizza', aliases: ['pizza', 'pizza pie'] },
    { food: 'hamburger', aliases: ['hamburger', 'cheeseburger', 'burger'] },
    { food: 'hot_dog', aliases: ['hotdog', 'hot dog', 'red hot'] },
    { food: 'ice_cream', aliases: ['ice cream', 'icecream', 'gelato'] },
    { food: 'donuts', aliases: ['doughnut', 'donut'] },
    { food: 'pretzel', aliases: ['pretzel'], calories: 338, displayName: 'Pretzel' },
    { food: 'bagel', aliases: ['bagel'], calories: 250, displayName: 'Bagel' },
    { food: 'guacamole', aliases: ['guacamole', 'avocado dip'] },
    { food: 'waffles', aliases: ['waffle'] },
    { food: 'pancakes', aliases: ['pancake', 'hotcake'] },
    { food: 'french_fries', aliases: ['french fries', 'french fry', 'fries'] },
    { food: 'nachos', aliases: ['nachos'] },
    { food: 'samosa', aliases: ['samosa'] },
    { food: 'sushi', aliases: ['sushi'] },
    { food: 'sashimi', aliases: ['sashimi'] },
    { food: 'ramen', aliases: ['ramen'] },
    { food: 'omelette', aliases: ['omelet', 'omelette'] },
    { food: 'macaroni_and_cheese', aliases: ['macaroni and cheese', 'mac and cheese'] },
    { food: 'spaghetti_carbonara', aliases: ['carbonara', 'spaghetti carbonara'] },
    { food: 'spaghetti_bolognese', aliases: ['bolognese', 'spaghetti bolognese'] },
    { food: 'apple_pie', aliases: ['apple pie'] },
    { food: 'cheesecake', aliases: ['cheesecake'] },
    { food: 'chocolate_cake', aliases: ['chocolate cake'] },
    { food: 'carrot_cake', aliases: ['carrot cake'] },
    { food: 'tiramisu', aliases: ['tiramisu'] },
    { food: 'baklava', aliases: ['baklava'] },
    { food: 'caesar_salad', aliases: ['caesar salad'] },
    { food: 'greek_salad', aliases: ['greek salad'] },
    { food: 'caprese_salad', aliases: ['caprese'] },
    { food: 'club_sandwich', aliases: ['club sandwich'] },
    { food: 'grilled_cheese_sandwich', aliases: ['grilled cheese'] },
    { food: 'breakfast_burrito', aliases: ['breakfast burrito', 'burrito'] },
    { food: 'chicken_wings', aliases: ['chicken wing', 'wings', 'buffalo wing'] },
    { food: 'steak', aliases: ['steak', 'sirloin'] },
    { food: 'tacos', aliases: ['taco'] },
    { food: 'lasagna', aliases: ['lasagna', 'lasagne'] },
    { food: 'garlic_bread', aliases: ['garlic bread'] },
    { food: 'fried_rice', aliases: ['fried rice'] },
    { food: 'chicken_curry', aliases: ['chicken curry', 'curry'] },
    { food: 'falafel', aliases: ['falafel'] },
    { food: 'onion_rings', aliases: ['onion ring'] },
    { food: 'hummus', aliases: ['hummus'] },
    { food: 'churros', aliases: ['churro'] },
    { food: 'cup_cakes', aliases: ['cupcake', 'cup cake'] },
    { food: 'poutine', aliases: ['poutine'] },
    { food: 'beignets', aliases: ['beignet'] },
    { food: 'dumplings', aliases: ['dumpling'] }
];

const CATEGORY_FALLBACKS = [
    { keyword: 'cake', displayName: 'Cake', calories: 350 },
    { keyword: 'pie', displayName: 'Pie', calories: 250 },
    { keyword: 'salad', displayName: 'Salad', calories: 140 },
    { keyword: 'sandwich', displayName: 'Sandwich', calories: 250 },
    { keyword: 'burger', displayName: 'Burger', calories: 295 },
    { keyword: 'pizza', displayName: 'Pizza', calories: 266 },
    { keyword: 'pasta', displayName: 'Pasta', calories: 190 },
    { keyword: 'noodle', displayName: 'Noodle Dish', calories: 170 },
    { keyword: 'rice', displayName: 'Rice Dish', calories: 160 },
    { keyword: 'curry', displayName: 'Curry', calories: 180 },
    { keyword: 'soup', displayName: 'Soup', calories: 100 },
    { keyword: 'bread', displayName: 'Bread Dish', calories: 280 },
    { keyword: 'fried', displayName: 'Fried Food', calories: 240 }
];

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragover');
    if (event.dataTransfer.files.length) {
        handleFile(event.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (event) => {
    if (event.target.files.length) {
        handleFile(event.target.files[0]);
    }
});

clearBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    clearPreview();
});

analyzeBtn.addEventListener('click', analyzeImage);

bootstrap();

function handleFile(file) {
    if (!file.type.match('image.*')) {
        showError('Please upload an image file in JPEG, PNG, or WEBP format.');
        return;
    }

    currentFile = file;
    const reader = new FileReader();

    reader.onload = (event) => {
        imagePreview.src = event.target.result;
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        hideError();
        resultsSection.classList.add('hidden');
        predictionDetailEl.classList.add('hidden');
        updateAnalyzeState();
    };

    reader.readAsDataURL(file);
}

function clearPreview() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    dropZone.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    resultsSection.classList.add('hidden');
    predictionDetailEl.classList.add('hidden');
    hideError();
    updateAnalyzeState();
}

async function analyzeImage() {
    if (!currentFile || !classifier) {
        return;
    }

    setLoadingState(true);
    hideError();
    resultsSection.classList.add('hidden');

    try {
        const predictions = await classifier.classify(imagePreview, 5);
        const analysis = mapPredictionToFood(predictions);

        if (!analysis) {
            throw new Error('No supported food was recognized. Try a clearer photo of pizza, burgers, desserts, sushi, fries, tacos, or similar foods.');
        }

        displayResults(analysis);
    } catch (error) {
        console.error('Analysis failed:', error);
        showError(error.message || 'Failed to analyze image. Please try again.');
    } finally {
        setLoadingState(false);
    }
}

function displayResults(result) {
    foodClassEl.textContent = result.food;
    foodCaloriesEl.textContent = typeof result.calories === 'number' ? `~${result.calories} kcal` : result.calories;
    foodConfidenceEl.textContent = `${result.confidence.toFixed(2)}%`;
    predictionDetailEl.textContent = result.detail;
    predictionDetailEl.classList.remove('hidden');
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setLoadingState(isLoading) {
    analyzeBtn.disabled = isLoading || !currentFile || !modelReady;
    if (isLoading) {
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
    } else {
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
    }
}

function updateAnalyzeState() {
    analyzeBtn.disabled = !currentFile || !modelReady;
}

function showError(message) {
    errorMessageEl.textContent = message;
    errorMessageEl.classList.remove('hidden');
}

function hideError() {
    errorMessageEl.classList.add('hidden');
}

async function bootstrap() {
    try {
        setStatus('Loading browser AI model...', 'loading');
        const [loadedModel, loadedCalorieMap] = await Promise.all([
            mobilenet.load({ version: 2, alpha: 1.0 }),
            loadCalorieMap()
        ]);

        classifier = loadedModel;
        calorieMap = loadedCalorieMap;
        modelReady = true;
        updateAnalyzeState();
        setStatus('Browser AI ready. Images stay on your device.', 'ready');
    } catch (error) {
        console.error('Model bootstrap failed:', error);
        setStatus('Model failed to load. Refresh and confirm internet access for CDN assets.', 'error');
        showError('The browser model could not be loaded.');
    }
}

async function loadCalorieMap() {
    const response = await fetch('calorie_map.json', { cache: 'no-store' });
    if (!response.ok) {
        throw new Error('Unable to load calorie map.');
    }
    return response.json();
}

function normalizeText(value) {
    return value.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function prettyFoodName(foodKey) {
    return foodKey.split('_').map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

function mapPredictionToFood(predictions) {
    for (const prediction of predictions) {
        const normalizedPrediction = normalizeText(prediction.className);

        for (const entry of FOOD_LABEL_MAP) {
            const matchedAlias = entry.aliases.find((alias) => normalizedPrediction.includes(normalizeText(alias)));
            if (!matchedAlias) {
                continue;
            }

            const calories = calorieMap[entry.food] ?? entry.calories ?? 'Estimate unavailable';
            const foodName = entry.displayName || prettyFoodName(entry.food);

            return {
                food: foodName,
                calories,
                confidence: prediction.probability * 100,
                detail: `Matched browser label "${prediction.className}" to ${foodName}. Calorie estimate is per 100g.`
            };
        }
    }

    for (const prediction of predictions) {
        const normalizedPrediction = normalizeText(prediction.className);

        for (const category of CATEGORY_FALLBACKS) {
            if (!normalizedPrediction.includes(category.keyword)) {
                continue;
            }

            return {
                food: category.displayName,
                calories: category.calories,
                confidence: prediction.probability * 100,
                detail: `Used a category fallback from browser label "${prediction.className}". Calorie estimate is approximate per 100g.`
            };
        }
    }

    return null;
}

function setStatus(message, tone) {
    statusBanner.textContent = message;
    statusBanner.className = `status-banner status-${tone}`;
}
