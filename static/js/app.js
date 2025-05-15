document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const modeBtns = document.querySelectorAll('.mode-btn');
    const uploadSections = document.querySelectorAll('.upload-section');
    const previewSections = document.querySelectorAll('.preview-section');
    const fileInputs = document.querySelectorAll('.file-input');
    const uploadLabels = document.querySelectorAll('.upload-label');
    const previewImage = document.getElementById('previewImage');
    const resultsDiv = document.querySelector('.results');
    const uploadBoxes = document.querySelectorAll('.upload-box');
    
    // Webcam elements
    const webcam = document.getElementById('webcam');
    const startAnalysisBtn = document.getElementById('startAnalysis');
    const stopAnalysisBtn = document.getElementById('stopAnalysis');
    
    // Webcam state
    let stream = null;
    let isAnalyzing = false;
    let analysisInterval = null;

    // Initialize webcam
    async function initWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcam.srcObject = stream;
        } catch (error) {
            console.error('Error accessing webcam:', error);
            resultsDiv.innerHTML = `<div class="error">Error: Could not access webcam. Please ensure you have granted camera permissions.</div>`;
        }
    }

    // Stop webcam
    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcam.srcObject = null;
            stream = null;
        }
    }

    // Start/Stop analysis
    startAnalysisBtn.addEventListener('click', () => {
        isAnalyzing = true;
        startAnalysisBtn.style.display = 'none';
        stopAnalysisBtn.style.display = 'block';
        
        // Start sending frames for analysis
        analysisInterval = setInterval(async () => {
            if (!isAnalyzing) return;
            
            try {
                // Capture frame from webcam
                const canvas = document.createElement('canvas');
                canvas.width = webcam.videoWidth;
                canvas.height = webcam.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(webcam, 0, 0);
                
                // Convert to blob
                const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
                
                // Send for analysis
                const formData = new FormData();
                formData.append('file', blob, 'frame.jpg');
                
                const response = await fetch('/analyze_image', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Analysis failed');
                }
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                console.error('Error analyzing frame:', error);
            }
        }, 1000); // Analyze every second
    });

    stopAnalysisBtn.addEventListener('click', () => {
        isAnalyzing = false;
        startAnalysisBtn.style.display = 'block';
        stopAnalysisBtn.style.display = 'none';
        clearInterval(analysisInterval);
    });

    // Mode switching
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            
            // Update active button
            modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Show corresponding upload section
            uploadSections.forEach(section => {
                section.classList.remove('active');
                if (section.dataset.mode === mode) {
                    section.classList.add('active');
                }
            });
            
            // Show corresponding preview section
            previewSections.forEach(section => {
                section.classList.remove('active');
                if (section.dataset.mode === mode) {
                    section.classList.add('active');
                }
            });
            
            // Reset preview and results
            resetPreview();
            resetResults();
            
            // Initialize webcam if video mode is selected
            if (mode === 'video') {
                initWebcam();
            } else {
                stopWebcam();
            }
        });
    });

    // File input handling
    fileInputs.forEach((input, index) => {
        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const mode = input.dataset.mode;
            
            // Reset previous preview
            resetPreview();
            
            // Show preview section
            previewSections.forEach(section => {
                section.classList.remove('active');
                if (section.dataset.mode === mode) {
                    section.classList.add('active');
                }
            });

            // Handle image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // Process the file
            processFile(file, mode);
        });
    });

    // Drag and drop handling
    uploadBoxes.forEach((box, index) => {
        box.addEventListener('dragover', (e) => {
            e.preventDefault();
            box.style.borderColor = 'var(--primary-color)';
        });

        box.addEventListener('dragleave', () => {
            box.style.borderColor = 'var(--border-color)';
        });

        box.addEventListener('drop', (e) => {
            e.preventDefault();
            box.style.borderColor = 'var(--border-color)';
            
            const file = e.dataTransfer.files[0];
            if (!file) return;

            const mode = box.dataset.mode;
            const fileInput = document.querySelector(`.file-input[data-mode="${mode}"]`);
            
            // Update file input
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            
            // Trigger change event
            fileInput.dispatchEvent(new Event('change'));
        });
    });

    // Reset preview
    function resetPreview() {
        previewImage.style.display = 'none';
        previewImage.src = '';
    }

    // Reset results
    function resetResults() {
        resultsDiv.innerHTML = '<div class="placeholder">Upload a file to see results</div>';
    }

    // Process file and get results
    async function processFile(file, mode) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            resultsDiv.innerHTML = '<div class="placeholder">Processing...</div>';
            
            const response = await fetch('/analyze_image', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        }
    }

    // Display results
    function displayResults(data) {
        let html = '<div class="results-content">';
        
        if (data.emotions) {
            html += `
                <div class="result-item">
                    <h3>Emotions</h3>
                    <ul>
                        ${Object.entries(data.emotions)
                            .map(([emotion, confidence]) => 
                                `<li>${emotion}: ${(confidence * 100).toFixed(2)}%</li>`
                            )
                            .join('')}
                    </ul>
                </div>
            `;
        }
        
        if (data.gender) {
            html += `
                <div class="result-item">
                    <h3>Gender</h3>
                    <p>${data.gender} (${(data.gender_confidence * 100).toFixed(2)}% confidence)</p>
                </div>
            `;
        }
        
        if (data.face_count) {
            html += `
                <div class="result-item">
                    <h3>Faces Detected</h3>
                    <p>${data.face_count}</p>
                </div>
            `;
        }
        
        html += '</div>';
        resultsDiv.innerHTML = html;
    }

    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        stopWebcam();
        if (analysisInterval) {
            clearInterval(analysisInterval);
        }
    });
}); 