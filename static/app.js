/**
 * Menu Taste Guide Interactive Interface
 * Connects to the FastAPI backend and demonstrates all core features
 */

// Global state
let currentRestaurant = null;
let currentMenu = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    // Show the first tab by default
    showTab('restaurant');

    // Set up file upload handling
    document.getElementById('menu-images').addEventListener('change', handleMenuImages);

    // Add enter key handlers
    document.getElementById('restaurant-name').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') findRestaurant();
    });

    document.getElementById('dish-name').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') analyzeTaste();
    });
});

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.add('hidden');
    });

    // Show selected tab
    document.getElementById(tabName + '-tab').classList.remove('hidden');

    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('bg-white', 'bg-opacity-30');
        btn.classList.add('bg-white', 'bg-opacity-20');
    });
}

// Restaurant search functionality
async function findRestaurant() {
    const name = document.getElementById('restaurant-name').value.trim();
    const city = document.getElementById('restaurant-city').value.trim();

    if (!name) {
        showError('Please enter a restaurant name');
        return;
    }

    const resultDiv = document.getElementById('restaurant-result');
    resultDiv.innerHTML = '<div class="flex items-center"><i class="fas fa-spinner loading mr-2"></i>Searching for restaurant...</div>';
    resultDiv.classList.remove('hidden');

    try {
        // Simulate API call (replace with actual API when keys are configured)
        await simulateDelay(1500);

        const mockResult = {
            place: {
                id: "mock-place-id",
                name: name,
                city: city || "Unknown City",
                address: "123 Main St, " + (city || "Seattle, WA"),
                rating: 4.3,
                review_count: 205,
                cuisine_tags: ["chinese", "sichuan"],
                price_level: 2
            },
            reviews_fetched: 120,
            cache_hit: false
        };

        currentRestaurant = mockResult.place;
        displayRestaurantResult(mockResult);

    } catch (error) {
        showError('Failed to find restaurant: ' + error.message);
        resultDiv.classList.add('hidden');
    }
}

function displayRestaurantResult(data) {
    const place = data.place;
    const html = `
        <div class="fade-in bg-gray-50 rounded-lg p-6">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <h3 class="text-xl font-bold text-gray-900">${place.name}</h3>
                    <p class="text-gray-600">${place.address}</p>
                    <div class="flex items-center mt-2">
                        <div class="flex text-yellow-400">
                            ${generateStars(place.rating)}
                        </div>
                        <span class="ml-2 text-sm text-gray-600">${place.rating} (${place.review_count} reviews)</span>
                    </div>
                </div>
                <div class="text-right">
                    <div class="flex items-center text-green-600">
                        <span class="text-2xl">${'$'.repeat(place.price_level)}</span>
                    </div>
                    <div class="flex flex-wrap gap-1 mt-2">
                        ${place.cuisine_tags.map(tag =>
                            `<span class="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">${tag}</span>`
                        ).join('')}
                    </div>
                </div>
            </div>

            <div class="grid md:grid-cols-2 gap-4 text-sm">
                <div class="flex items-center text-gray-600">
                    <i class="fas fa-comments mr-2 text-blue-500"></i>
                    ${data.reviews_fetched} reviews analyzed
                </div>
                <div class="flex items-center text-gray-600">
                    <i class="fas fa-clock mr-2 text-green-500"></i>
                    ${data.cache_hit ? 'Cached data' : 'Fresh data'}
                </div>
            </div>

            <div class="mt-4 pt-4 border-t border-gray-200">
                <button onclick="showTab('menu')" class="bg-green-600 text-white px-4 py-2 rounded mr-2 hover:bg-green-700 transition">
                    <i class="fas fa-camera mr-2"></i>Scan Menu
                </button>
                <button onclick="showTab('taste')" class="bg-purple-600 text-white px-4 py-2 rounded mr-2 hover:bg-purple-700 transition">
                    <i class="fas fa-brain mr-2"></i>Analyze Dishes
                </button>
                <button onclick="showTab('combo')" class="bg-orange-600 text-white px-4 py-2 rounded hover:bg-orange-700 transition">
                    <i class="fas fa-utensils mr-2"></i>Get Combos
                </button>
            </div>
        </div>
    `;

    document.getElementById('restaurant-result').innerHTML = html;

    // Auto-fill restaurant name in other tabs
    document.getElementById('dish-restaurant').value = place.name;
}

// Menu image handling
function handleMenuImages(event) {
    const files = event.target.files;
    if (files.length === 0) return;

    const previewDiv = document.getElementById('menu-preview');
    previewDiv.innerHTML = '';
    previewDiv.classList.remove('hidden');

    // Show image previews
    for (let i = 0; i < Math.min(files.length, 3); i++) {
        const file = files[i];
        const reader = new FileReader();

        reader.onload = function(e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'w-32 h-32 object-cover rounded-lg border border-gray-300';
            previewDiv.appendChild(img);
        };

        reader.readAsDataURL(file);
    }

    // Process the menu
    processMenuImages(files);
}

async function processMenuImages(files) {
    const resultDiv = document.getElementById('menu-result');
    resultDiv.innerHTML = '<div class="flex items-center mt-4"><i class="fas fa-spinner loading mr-2"></i>Processing menu images with OCR...</div>';

    try {
        await simulateDelay(2000);

        // Mock OCR results
        const mockMenuItems = [
            { name: "Kung Pao Chicken", price: 16.95, section: "Main Dishes", confidence: 0.92 },
            { name: "Mapo Tofu", price: 14.95, section: "Main Dishes", confidence: 0.88 },
            { name: "Dry-Fried Green Beans", price: 12.95, section: "Vegetables", confidence: 0.95 },
            { name: "Dan Dan Noodles", price: 13.95, section: "Noodles", confidence: 0.89 },
            { name: "Hot and Sour Soup", price: 8.95, section: "Soups", confidence: 0.91 },
            { name: "Steamed Rice", price: 3.95, section: "Sides", confidence: 0.97 }
        ];

        currentMenu = mockMenuItems;
        displayMenuResults(mockMenuItems);

    } catch (error) {
        showError('Failed to process menu images: ' + error.message);
    }
}

function displayMenuResults(menuItems) {
    const html = `
        <div class="fade-in">
            <h3 class="text-lg font-bold mb-4 flex items-center">
                <i class="fas fa-list-ul text-green-600 mr-2"></i>
                Extracted Menu Items (${menuItems.length})
            </h3>

            <div class="grid gap-3">
                ${menuItems.map(item => `
                    <div class="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <div class="flex-1">
                            <div class="font-medium">${item.name}</div>
                            <div class="text-sm text-gray-600">${item.section}</div>
                        </div>
                        <div class="text-right">
                            <div class="font-bold text-green-600">$${item.price}</div>
                            <div class="text-xs text-gray-500">${Math.round(item.confidence * 100)}% confidence</div>
                        </div>
                        <button onclick="analyzeSpecificTaste('${item.name}')"
                                class="ml-3 bg-purple-600 text-white px-3 py-1 rounded text-sm hover:bg-purple-700 transition">
                            Analyze
                        </button>
                    </div>
                `).join('')}
            </div>

            <div class="mt-6 p-4 bg-blue-50 rounded-lg">
                <h4 class="font-bold text-blue-800 mb-2">Next Steps:</h4>
                <div class="grid md:grid-cols-2 gap-2 text-sm">
                    <div class="flex items-center text-blue-700">
                        <i class="fas fa-brain mr-2"></i>Click "Analyze" to get taste descriptions
                    </div>
                    <div class="flex items-center text-blue-700">
                        <i class="fas fa-utensils mr-2"></i>Go to Combos for meal recommendations
                    </div>
                </div>
            </div>
        </div>
    `;

    document.getElementById('menu-result').innerHTML = html;
}

// Taste analysis functionality
async function analyzeTaste() {
    const dishName = document.getElementById('dish-name').value.trim();
    const restaurant = document.getElementById('dish-restaurant').value.trim();

    if (!dishName) {
        showError('Please enter a dish name');
        return;
    }

    await analyzeSpecificTaste(dishName, restaurant);
}

async function analyzeSpecificTaste(dishName, restaurant = null) {
    const resultDiv = document.getElementById('taste-result');
    resultDiv.innerHTML = '<div class="flex items-center"><i class="fas fa-spinner loading mr-2"></i>Analyzing taste profile...</div>';

    // Fill in the dish name if called from menu
    document.getElementById('dish-name').value = dishName;
    if (restaurant) {
        document.getElementById('dish-restaurant').value = restaurant;
    }

    try {
        await simulateDelay(2000);

        // Mock taste analysis based on dish name
        const tasteData = getMockTasteData(dishName);
        displayTasteResults(tasteData);

    } catch (error) {
        showError('Failed to analyze taste: ' + error.message);
    }
}

function getMockTasteData(dishName) {
    const dishLower = dishName.toLowerCase();

    // Mock data based on common dishes
    if (dishLower.includes('mapo') || dishLower.includes('tofu')) {
        return {
            dish_name: dishName,
            bullets: [
                "Very spicy with intense heat that may overwhelm sensitive palates",
                "Features numbing Sichuan peppercorns that create a tingling sensation",
                "Silky soft tofu in rich, oily red sauce",
                "High umami from ground pork and fermented bean paste",
                "Generous portion size - good for sharing"
            ],
            aspects: {
                spice: 3,
                heat_type: ["peppercorn"],
                texture: ["silky"],
                richness: "heavy",
                portion: "large"
            },
            pairing: "Pairs well with steamed rice to balance the heat and cold beer to cool the palate",
            confidence: "high",
            sources: ["120+ Google reviews", "85 Yelp reviews"],
            review_snippets_used: 8
        };
    } else if (dishLower.includes('kung pao')) {
        return {
            dish_name: dishName,
            bullets: [
                "Medium spice level with balanced heat",
                "Crispy chicken pieces with crunchy peanuts",
                "Sweet and savory sauce with garlic notes",
                "Good texture contrast between soft and crunchy elements",
                "Not too oily, well-balanced flavors"
            ],
            aspects: {
                spice: 2,
                heat_type: ["chili"],
                texture: ["crispy", "crunchy"],
                richness: "medium",
                allergens: ["peanut"]
            },
            pairing: "Goes well with steamed rice and light vegetables",
            confidence: "high",
            sources: ["95+ Google reviews"],
            review_snippets_used: 6
        };
    } else {
        return {
            dish_name: dishName,
            bullets: [
                "Flavor profile varies by preparation style",
                "Typically features balanced seasoning",
                "Texture depends on cooking method used",
                "Portion size is usually moderate"
            ],
            aspects: {
                spice: 1,
                texture: ["varied"],
                richness: "medium"
            },
            pairing: "Pairs well with rice and complementary dishes",
            confidence: "medium",
            sources: ["Limited review data"],
            review_snippets_used: 2
        };
    }
}

function displayTasteResults(data) {
    const spiceIndicators = generateSpiceIndicators(data.aspects.spice);
    const confidenceColor = data.confidence === 'high' ? 'green' : data.confidence === 'medium' ? 'yellow' : 'red';

    const html = `
        <div class="fade-in bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-6">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <h3 class="text-xl font-bold text-gray-900">${data.dish_name}</h3>
                    <div class="flex items-center mt-2">
                        <span class="text-sm text-gray-600 mr-3">Spice Level:</span>
                        ${spiceIndicators}
                        <span class="ml-2 text-sm text-gray-700">${getSpiceLabel(data.aspects.spice)}</span>
                    </div>
                </div>
                <div class="text-right">
                    <div class="flex items-center">
                        <span class="text-sm text-gray-600 mr-2">Confidence:</span>
                        <span class="bg-${confidenceColor}-100 text-${confidenceColor}-800 px-2 py-1 rounded text-sm font-medium">
                            ${data.confidence.toUpperCase()}
                        </span>
                    </div>
                </div>
            </div>

            <div class="mb-6">
                <h4 class="font-bold text-gray-800 mb-3 flex items-center">
                    <i class="fas fa-list-ul mr-2 text-purple-600"></i>Taste Profile
                </h4>
                <ul class="space-y-2">
                    ${data.bullets.map(bullet =>
                        `<li class="flex items-start"><i class="fas fa-check-circle text-green-500 mr-2 mt-1 text-sm"></i><span class="text-gray-700">${bullet}</span></li>`
                    ).join('')}
                </ul>
            </div>

            <div class="grid md:grid-cols-2 gap-6 mb-6">
                <div>
                    <h4 class="font-bold text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-palette mr-2 text-blue-600"></i>Taste Aspects
                    </h4>
                    <div class="space-y-2 text-sm">
                        ${Object.entries(data.aspects).map(([key, value]) => {
                            if (value && value.length > 0) {
                                const displayValue = Array.isArray(value) ? value.join(', ') : value;
                                return `<div class="flex justify-between"><span class="capitalize text-gray-600">${key.replace('_', ' ')}:</span><span class="font-medium">${displayValue}</span></div>`;
                            }
                            return '';
                        }).filter(item => item).join('')}
                    </div>
                </div>

                <div>
                    <h4 class="font-bold text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-utensils mr-2 text-orange-600"></i>Pairing Suggestion
                    </h4>
                    <p class="text-gray-700 text-sm bg-orange-50 p-3 rounded">${data.pairing}</p>
                </div>
            </div>

            <div class="border-t border-gray-200 pt-4">
                <div class="flex justify-between items-center text-sm text-gray-600">
                    <div class="flex items-center">
                        <i class="fas fa-chart-bar mr-2"></i>
                        Based on ${data.review_snippets_used} review snippets
                    </div>
                    <div class="flex items-center">
                        <i class="fas fa-database mr-2"></i>
                        Sources: ${data.sources.join(', ')}
                    </div>
                </div>
            </div>
        </div>
    `;

    document.getElementById('taste-result').innerHTML = html;
}

// Combo recommendations functionality
async function getComboRecommendations() {
    const partySize = document.getElementById('party-size').value;
    const budget = document.getElementById('budget').value;
    const spicePreference = document.getElementById('spice-preference').value;

    const dietaryConstraints = Array.from(document.querySelectorAll('.dietary-constraint:checked'))
        .map(cb => cb.value);

    const resultDiv = document.getElementById('combo-result');
    resultDiv.innerHTML = '<div class="flex items-center"><i class="fas fa-spinner loading mr-2"></i>Analyzing combinations and generating recommendations...</div>';

    try {
        await simulateDelay(2500);

        const mockCombos = generateMockCombos(partySize, budget, spicePreference, dietaryConstraints);
        displayComboResults(mockCombos, partySize, budget);

    } catch (error) {
        showError('Failed to generate combo recommendations: ' + error.message);
    }
}

function generateMockCombos(partySize, budget, spicePreference, dietaryConstraints) {
    const combos = [
        {
            items: [
                { name: "Mapo Tofu", role: "protein", spice_level: 3, price: 14.95 },
                { name: "Dry-Fried Green Beans", role: "vegetable", spice_level: 0, price: 12.95 },
                { name: "Steamed Rice", role: "carb", spice_level: 0, price: 3.95 }
            ],
            score: 87,
            rationale: "Balanced combination with good spice variety and diverse textures. The rich, spicy tofu is perfectly complemented by crispy vegetables and neutral rice.",
            estimated_total: 31.85,
            meets_dietary_requirements: !dietaryConstraints.includes('vegan'),
            balance_matrix: {
                spice_range: [0, 3],
                texture_variety: ["silky", "crispy", "fluffy"],
                richness_variety: ["heavy", "light", "light"]
            }
        },
        {
            items: [
                { name: "Kung Pao Chicken", role: "protein", spice_level: 2, price: 16.95 },
                { name: "Hot and Sour Soup", role: "soup", spice_level: 1, price: 8.95 },
                { name: "Dan Dan Noodles", role: "carb", spice_level: 2, price: 13.95 }
            ],
            score: 82,
            rationale: "Well-rounded meal with moderate spice levels and varied cooking styles. Includes protein, soup, and noodles for a complete dining experience.",
            estimated_total: 39.85,
            meets_dietary_requirements: !dietaryConstraints.includes('vegetarian'),
            balance_matrix: {
                spice_range: [1, 2],
                texture_variety: ["crispy", "liquid", "chewy"],
                richness_variety: ["medium", "light", "medium"]
            }
        }
    ];

    // Filter by dietary constraints and spice preference
    return combos.filter(combo => {
        const maxSpice = Math.max(...combo.items.map(item => item.spice_level));
        return maxSpice <= parseInt(spicePreference) && combo.meets_dietary_requirements;
    });
}

function displayComboResults(combos, partySize, budget) {
    if (combos.length === 0) {
        document.getElementById('combo-result').innerHTML = `
            <div class="fade-in bg-yellow-50 border border-yellow-200 rounded-lg p-6">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-triangle text-yellow-600 mr-3"></i>
                    <div>
                        <h3 class="font-bold text-yellow-800">No suitable combinations found</h3>
                        <p class="text-yellow-700 text-sm mt-1">Try adjusting your preferences or dietary restrictions.</p>
                    </div>
                </div>
            </div>
        `;
        return;
    }

    const html = `
        <div class="fade-in">
            <div class="flex justify-between items-center mb-6">
                <h3 class="text-lg font-bold flex items-center">
                    <i class="fas fa-magic text-orange-600 mr-2"></i>
                    Recommended Combinations
                </h3>
                <div class="text-sm text-gray-600">
                    For ${partySize} ${partySize == 1 ? 'person' : 'people'}${budget ? `, $${budget} budget` : ''}
                </div>
            </div>

            <div class="space-y-6">
                ${combos.map((combo, index) => `
                    <div class="taste-card bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                        <div class="flex justify-between items-start mb-4">
                            <div>
                                <h4 class="font-bold text-lg text-gray-900">Option ${index + 1}</h4>
                                <div class="flex items-center mt-1">
                                    <span class="bg-green-100 text-green-800 px-2 py-1 rounded text-sm font-medium mr-2">
                                        Score: ${combo.score}
                                    </span>
                                    <span class="text-lg font-bold text-gray-900">$${combo.estimated_total.toFixed(2)}</span>
                                </div>
                            </div>
                            <div class="text-right">
                                <div class="text-sm text-gray-600 mb-1">Max Spice:</div>
                                ${generateSpiceIndicators(Math.max(...combo.items.map(item => item.spice_level)))}
                            </div>
                        </div>

                        <div class="grid gap-3 mb-4">
                            ${combo.items.map(item => `
                                <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
                                    <div class="flex-1">
                                        <div class="font-medium">${item.name}</div>
                                        <div class="text-sm text-gray-600 capitalize">${item.role}</div>
                                    </div>
                                    <div class="flex items-center">
                                        ${generateSpiceIndicators(item.spice_level, true)}
                                        <span class="ml-3 font-medium text-gray-900">$${item.price}</span>
                                    </div>
                                </div>
                            `).join('')}
                        </div>

                        <div class="bg-blue-50 rounded-lg p-4 mb-4">
                            <h5 class="font-medium text-blue-900 mb-2">Why this works:</h5>
                            <p class="text-blue-800 text-sm">${combo.rationale}</p>
                        </div>

                        <div class="grid md:grid-cols-3 gap-4 text-sm">
                            <div>
                                <div class="font-medium text-gray-700 mb-1">Spice Range</div>
                                <div class="text-gray-600">${combo.balance_matrix.spice_range[0]} - ${combo.balance_matrix.spice_range[1]}</div>
                            </div>
                            <div>
                                <div class="font-medium text-gray-700 mb-1">Textures</div>
                                <div class="text-gray-600">${combo.balance_matrix.texture_variety.join(', ')}</div>
                            </div>
                            <div>
                                <div class="font-medium text-gray-700 mb-1">Richness</div>
                                <div class="text-gray-600">${combo.balance_matrix.richness_variety.join(', ')}</div>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>

            <div class="mt-6 p-4 bg-green-50 rounded-lg">
                <h4 class="font-bold text-green-800 mb-2">Pro Tips:</h4>
                <ul class="text-green-700 text-sm space-y-1">
                    <li class="flex items-center"><i class="fas fa-lightbulb mr-2"></i>Order dishes to arrive together for the best experience</li>
                    <li class="flex items-center"><i class="fas fa-users mr-2"></i>These portions are estimated for ${partySize} ${partySize == 1 ? 'person' : 'people'}</li>
                    <li class="flex items-center"><i class="fas fa-fire mr-2"></i>Start with milder dishes if you're sensitive to spice</li>
                </ul>
            </div>
        </div>
    `;

    document.getElementById('combo-result').innerHTML = html;
}

// Demo data loading functions
function loadDemoRestaurant() {
    document.getElementById('restaurant-name').value = 'Szechuan Chef';
    document.getElementById('restaurant-city').value = 'Seattle, WA';
    findRestaurant();
}

function loadDemoMenu() {
    showTab('menu');
    // Simulate loading demo menu
    const mockMenuItems = [
        { name: "Kung Pao Chicken", price: 16.95, section: "Main Dishes", confidence: 0.92 },
        { name: "Mapo Tofu", price: 14.95, section: "Main Dishes", confidence: 0.88 },
        { name: "Dry-Fried Green Beans", price: 12.95, section: "Vegetables", confidence: 0.95 }
    ];
    currentMenu = mockMenuItems;
    displayMenuResults(mockMenuItems);
}

function loadDemoTaste() {
    showTab('taste');
    document.getElementById('dish-name').value = 'Mapo Tofu';
    document.getElementById('dish-restaurant').value = 'Szechuan Chef';
    analyzeTaste();
}

function loadDemoCombo() {
    showTab('combo');
    document.getElementById('party-size').value = '2';
    document.getElementById('budget').value = '50';
    document.getElementById('spice-preference').value = '2';
    getComboRecommendations();
}

// Utility functions
function generateStars(rating) {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);

    return '★'.repeat(fullStars) +
           (hasHalfStar ? '☆' : '') +
           '☆'.repeat(emptyStars);
}

function generateSpiceIndicators(level, small = false) {
    const size = small ? 'w-2 h-2' : 'w-3 h-3';
    let indicators = '';
    for (let i = 0; i < 3; i++) {
        const active = i < level;
        indicators += `<span class="spice-indicator ${size} rounded-full ${active ? 'spice-active' : 'spice-inactive'}"></span>`;
    }
    return indicators;
}

function getSpiceLabel(level) {
    const labels = ['None', 'Mild', 'Medium', 'Very Spicy'];
    return labels[level] || 'Unknown';
}

function showError(message) {
    // Create a simple toast notification
    const toast = document.createElement('div');
    toast.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
    toast.innerHTML = `<i class="fas fa-exclamation-circle mr-2"></i>${message}`;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 5000);
}

function simulateDelay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}