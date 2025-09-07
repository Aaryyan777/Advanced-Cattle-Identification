
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const preview = document.getElementById('preview');
    const resultsDiv = document.getElementById('results');
    const spinner = document.getElementById('spinner');

    // Trigger file input when the visible label is clicked
    const uploadLabel = document.querySelector('.upload-label');
    if (uploadLabel) {
        uploadLabel.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent form submission if it's a button
            imageInput.click();
        });
    }

    imageInput.addEventListener('change', () => {
        const file = imageInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.classList.remove('hidden');
                resultsDiv.innerHTML = ''; // Clear previous results
            };
            reader.readAsDataURL(file);
            // Automatically submit the form once an image is selected
            form.dispatchEvent(new Event('submit', { cancelable: true }));
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!imageInput.files || imageInput.files.length === 0) {
            alert('Please select an image first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', imageInput.files[0]);

        spinner.classList.remove('hidden');
        resultsDiv.innerHTML = '';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            displayResults(data);
        } catch (error) {
            resultsDiv.innerHTML = `<div class="result-card error"><h3>SYSTEM_ERROR</h3><p>${error.toString()}</p></div>`;
            console.error('Error:', error);
        } finally {
            spinner.classList.add('hidden');
        }
    });

    function displayResults(data) {
        resultsDiv.innerHTML = ''; // Clear previous results

        if (data.error) {
            resultsDiv.innerHTML = `<div class="result-card error"><h3>ERROR</h3><p>${data.error}</p></div>`;
            return;
        }

        if (data.is_cattle === false) {
            const card = document.createElement('div');
            card.className = 'result-card not-cattle';
            card.innerHTML = `
                <h3>FILTER_RESULT: NEGATIVE</h3>
                <p>INPUT_IMG_REJECTED</p>
                <p>CLASSIFICATION: ${data.filter_result.top_prediction_label.replace('a photo of a', '').toUpperCase()}</p>
                <p>CONFIDENCE: ${parseFloat(data.filter_result.top_prediction_confidence).toFixed(4)}</p>
            `;
            resultsDiv.appendChild(card);
        } else {
            let predictionsHtml = data.predictions.map(p => `
                <div class="prediction-item">
                    <span class="label">${p.label.toUpperCase()}</span>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar" style="width: ${p.confidence * 100}%;"></div>
                    </div>
                    <span class="score">${(p.confidence * 100).toFixed(2)}%</span>
                </div>
            `).join('');

            const card = document.createElement('div');
            card.className = 'result-card is-cattle';
            card.innerHTML = `
                <h3>FILTER_RESULT: POSITIVE</h3>
                <p>INPUT_IMG_ACCEPTED</p>
                <p>ROUTING_TO_EXPERT_MODEL...</p>
                <br>
                <h3>EXPERT_ANALYSIS:</h3>
                ${predictionsHtml}
            `;
            resultsDiv.appendChild(card);
        }
    }
});
