// Document Scanner & Enhancer - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const rawImage = document.getElementById('rawImage');
    const enhancedImage = document.getElementById('enhancedImage');
    const errorDiv = document.getElementById('error');

    // Click to upload
    dropzone.addEventListener('click', function() {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop events
    dropzone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Handle file processing
    function handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file (PNG, JPG, etc.)');
            return;
        }

        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            showError('File size must be less than 10MB');
            return;
        }

        // Hide previous results and errors
        results.classList.add('hidden');
        errorDiv.classList.add('hidden');

        // Show raw photo preview
        const objectUrl = URL.createObjectURL(file);
        rawImage.src = objectUrl;
        rawImage.onload = function() {
            URL.revokeObjectURL(objectUrl);
        };

        // Show loading indicator
        loading.classList.remove('hidden');

        // Upload and process file
        uploadAndProcess(file);
    }

    // Upload file to server
    function uploadAndProcess(file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/scan', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.detail || 'Processing failed'); });
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                // Display enhanced image
                const base64Image = `data:image/jpeg;base64,${data.enhanced_image}`;
                enhancedImage.src = base64Image;

                // Hide loading, show results
                loading.classList.add('hidden');
                results.classList.remove('hidden');
            } else {
                throw new Error('Unexpected response from server');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loading.classList.add('hidden');
            showError(error.message || 'Failed to process image. Please try again.');
        });
    }

    // Show error message
    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
    }
});
