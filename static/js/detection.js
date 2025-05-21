// detection.js - File detection page functionality

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file');
    const previewContainer = document.getElementById('previewContainer');
    const previewContent = document.getElementById('previewContent');
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const resultsSection = document.getElementById('resultsSection');
    const mediaResult = document.getElementById('mediaResult');
    const detectionStats = document.getElementById('detectionStats');
    const viewDetailedBtn = document.getElementById('viewDetailedBtn');
    const resetBtn = document.getElementById('resetBtn');
    
    // Add progress container
    const progressContainerHtml = `
        <div class="progress-container mt-3 d-none" id="progressContainer">
            <h4>Processing...</h4>
            <div class="progress">
                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                    id="progressBar" role="progressbar" 
                    aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" 
                    style="width: 0%">0%</div>
            </div>
            <p id="progressStatus" class="text-center mt-2">Initializing...</p>
        </div>
    `;
    
    if (uploadForm) {
        // Add progress container after the form
        uploadForm.insertAdjacentHTML('afterend', progressContainerHtml);
    }
    
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressStatus = document.getElementById('progressStatus');
    
    // Reset button functionality
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            // Hide results section
            resultsSection.classList.add('d-none');
            
            // Clear file input
            if (fileInput) {
                fileInput.value = '';
            }
            
            // Hide preview
            previewContainer.classList.add('d-none');
            previewContent.innerHTML = '';
            
            // Reset progress
            progressContainer.classList.add('d-none');
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            
            // Enable upload button
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="fas fa-upload me-1"></i> Upload and Detect';
        });
    }
    
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            
            if (file) {
                const fileType = file.type;
                previewContainer.classList.remove('d-none');
                
                // Clear previous preview
                previewContent.innerHTML = '';
                
                if (fileType.startsWith('image/')) {
                    // Preview image
                    const img = document.createElement('img');
                    img.classList.add('preview-image');
                    img.src = URL.createObjectURL(file);
                    previewContent.appendChild(img);
                } else if (fileType.startsWith('video/')) {
                    // Preview video
                    const video = document.createElement('video');
                    video.classList.add('preview-video');
                    video.controls = true;
                    video.src = URL.createObjectURL(file);
                    previewContent.appendChild(video);
                }
            }
        });
    }
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            // Show progress container
            progressContainer.classList.remove('d-none');
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            progressStatus.textContent = 'Uploading file...';
            
            // Hide results section if visible
            resultsSection.classList.add('d-none');
            
            // Disable upload button
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Processing...';
            
            // Send AJAX request
            fetch('/file_detection', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    // Show error
                    progressStatus.textContent = 'Error: ' + data.error;
                    progressBar.classList.remove('bg-info');
                    progressBar.classList.add('bg-danger');
                    
                    // Re-enable upload button
                    uploadBtn.disabled = false;
                    uploadBtn.innerHTML = '<i class="fas fa-upload me-1"></i> Upload and Detect';
                } else {
                    // Start polling for progress
                    progressStatus.textContent = 'Processing...';
                    pollProgress(data.detection_id);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                progressStatus.textContent = 'Error: ' + error.message;
                progressBar.classList.remove('bg-info');
                progressBar.classList.add('bg-danger');
                
                // Re-enable upload button
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = '<i class="fas fa-upload me-1"></i> Upload and Detect';
            });
        });
    }
    
    function pollProgress(detectionId) {
        const interval = setInterval(function() {
            fetch(`/detection_progress/${detectionId}`)
                .then(response => response.json())
                .then(data => {
                    // Update progress bar
                    const progress = data.progress || 0;
                    progressBar.style.width = `${progress}%`;
                    progressBar.textContent = `${progress}%`;
                    progressStatus.textContent = `Processing: ${progress}% complete`;
                    
                    if (data.status === 'completed') {
                        clearInterval(interval);
                        progressStatus.textContent = 'Processing complete!';
                        progressBar.classList.remove('progress-bar-animated');
                        
                        // Load and display results
                        loadDetectionResults(detectionId);
                        
                        // Re-enable upload button
                        uploadBtn.disabled = false;
                        uploadBtn.innerHTML = '<i class="fas fa-upload me-1"></i> Upload and Detect';
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        progressStatus.textContent = 'Error: ' + (data.error_message || 'Unknown error');
                        progressBar.classList.remove('bg-info');
                        progressBar.classList.add('bg-danger');
                        
                        // Re-enable upload button
                        uploadBtn.disabled = false;
                        uploadBtn.innerHTML = '<i class="fas fa-upload me-1"></i> Upload and Detect';
                    }
                })
                .catch(error => {
                    console.error('Error polling progress:', error);
                });
        }, 1000); // Poll every second
    }
    
    function loadDetectionResults(detectionId) {
        fetch(`/detection_result/${detectionId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error('Error loading results:', data.error);
                    return;
                }
                
                // Show results section
                resultsSection.classList.remove('d-none');
                
                // Display media result
                mediaResult.innerHTML = '';
                if (data.detection_type === 'image') {
                    const img = document.createElement('img');
                    img.classList.add('result-image');
                    img.src = data.result_url;
                    mediaResult.appendChild(img);
                } else if (data.detection_type === 'video') {
                    const video = document.createElement('video');
                    video.classList.add('result-video');
                    video.controls = true;
                    video.src = data.result_url;
                    mediaResult.appendChild(video);
                }
                
                // Display detection stats
                const statsHtml = `
                    <div class="row">
                        <div class="col-md-6">
                            <div class="stat-item">
                                <div class="stat-icon">
                                    <i class="fas fa-bed"></i>
                                </div>
                                <div>
                                    <div>Drowsiness Detections</div>
                                    <div class="stat-value">${data.drowsy_count}</div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="stat-item">
                                <div class="stat-icon">
                                    <i class="fas fa-tired"></i>
                                </div>
                                <div>
                                    <div>Yawn Detections</div>
                                    <div class="stat-value">${data.yawn_count}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="stat-item">
                                <div class="stat-icon">
                                    <i class="fas fa-arrows-alt"></i>
                                </div>
                                <div>
                                    <div>Head Movement Detections</div>
                                    <div class="stat-value">${data.head_movement_count}</div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="stat-item">
                                <div class="stat-icon">
                                    <i class="fas fa-tachometer-alt"></i>
                                </div>
                                <div>
                                    <div>Average FPS</div>
                                    <div class="stat-value">${data.avg_fps.toFixed(2)}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                detectionStats.innerHTML = statsHtml;
                
                // Update detailed results button link
                if (viewDetailedBtn) {
                    viewDetailedBtn.href = `/view_result/${detectionId}`;
                }
                
                // Scroll to results section
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }
});
