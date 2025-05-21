document.addEventListener('DOMContentLoaded', function() {
    const startWebcamBtn = document.getElementById('startWebcamBtn');
    const stopWebcamBtn = document.getElementById('stopWebcamBtn');
    const videoStream = document.getElementById('videoStream');
    const cameraIcon = document.getElementById('cameraIcon');
    const webcamStatus = document.getElementById('webcamStatus');
    const statsContainer = document.getElementById('statsContainer');
    
    let isDetecting = false;
    let detectionId = null;
    let recordingStartTime = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let statsData = {
        drowsy_count: 0,
        yawn_count: 0,
        head_count: 0
    };
    
    if (startWebcamBtn) {
        startWebcamBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Show loading status
            webcamStatus.textContent = "Starting webcam stream...";
            if (cameraIcon) {
                cameraIcon.style.display = 'none';
            }
            
            // Create detection record
            fetch('/start_webcam_detection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                detectionId = data.detection_id;
                
                // Update video source with detection ID
                videoStream.src = `/video_feed/${detectionId}`;
                
                // Show video stream
                videoStream.classList.remove('d-none');
                // statsContainer.classList.remove('d-none');
                
                
                // Start detection
                isDetecting = true;
                webcamStatus.textContent = "Detecting...";
                
                // Setup recording - using MediaRecorder API for better quality
                setupMediaRecording();
                
                // Show stop button
                stopWebcamBtn.classList.remove('d-none');
                startWebcamBtn.classList.add('d-none');
                
                // Start updating stats
                startStatsUpdate();
            })
            .catch(error => {
                console.error('Error starting detection:', error);
                webcamStatus.textContent = "Error starting detection. Please try again.";
                webcamStatus.classList.add('text-danger');
            });
        });
    }
    
    if (stopWebcamBtn) {
        stopWebcamBtn.addEventListener('click', function() {
            stopDetection();
        });
    }
    
    function setupMediaRecording() {
        // Use MediaRecorder to record the video stream
        recordingStartTime = Date.now();
        
        try {
            // Create a canvas to capture the video stream
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 640;
            canvas.height = 480;
            
            // Create a stream from the canvas
            const stream = canvas.captureStream(30); // 30 FPS
            
            // Create MediaRecorder
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp9',
                videoBitsPerSecond: 2500000 // 2.5 Mbps
            });
            
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            
            mediaRecorder.start(1000); // Collect data every second
            
            // Draw frames from video element to canvas
            const drawInterval = setInterval(() => {
                if (!isDetecting) {
                    clearInterval(drawInterval);
                    return;
                }
                
                if (videoStream.complete && videoStream.naturalHeight !== 0) {
                    ctx.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
                }
            }, 33); // ~30 FPS
            
        } catch (error) {
            console.error('Error setting up media recording:', error);
            // Fallback to frame-by-frame recording
            setupFrameRecording();
        }
    }
    
    function setupFrameRecording() {
        // Fallback recording method using frame capture
        recordingStartTime = Date.now();
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 640;
        canvas.height = 480;
        
        const recordInterval = setInterval(() => {
            if (!isDetecting) {
                clearInterval(recordInterval);
                return;
            }
            
            // Draw frame from video to canvas
            if (videoStream.complete && videoStream.naturalHeight !== 0) {
                ctx.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
                
                // Save frame as blob
                canvas.toBlob(blob => {
                    if (blob) {
                        recordedChunks.push(blob);
                    }
                }, 'image/jpeg', 0.85);
            }
        }, 100); // 10 FPS
    }
    
    function startStatsUpdate() {
        // Update stats from server periodically
        const statsInterval = setInterval(() => {
            if (!isDetecting) {
                clearInterval(statsInterval);
                return;
            }
            
            // Get stats from server
            fetch(`/check_processing_status/${detectionId}`)
                .then(response => response.json())
                .then(data => {
                    // Store the latest stats
                    statsData.drowsy_count = data.drowsy_count || 0;
                    statsData.yawn_count = data.yawn_count || 0;
                    statsData.head_count = data.head_movement_count || 0;
                    
                    // Update UI
                    document.getElementById('drowsyCount').textContent = statsData.drowsy_count;
                    document.getElementById('yawnCount').textContent = statsData.yawn_count;
                    document.getElementById('headCount').textContent = statsData.head_count;
                })
                .catch(error => {
                    console.error('Error fetching stats:', error);
                });
        }, 1000); // Update every second
    }
    
    function stopDetection() {
        isDetecting = false;
        webcamStatus.textContent = "Processing recording...";
        
        // Hide video stream
        videoStream.classList.add('d-none');
        videoStream.src = '';
        if (cameraIcon) {
            cameraIcon.style.display = 'block';
        }
        
        // Send stop request to backend
        fetch('/stop_webcam_detection', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                detection_id: detectionId
            })
        });
        
        // Stop media recorder if active
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            
            // Wait a bit for the last data
            setTimeout(() => {
                processRecording();
            }, 500);
        } else {
            // Process frame recording
            processRecording();
        }
    }
    
    function processRecording() {
        // Create video from recorded chunks
        let blob;
        
        try {
            if (mediaRecorder) {
                // For MediaRecorder, chunks are already video format
                blob = new Blob(recordedChunks, { type: 'video/webm' });
            } else {
                // For frame recording, convert to video format
                blob = new Blob(recordedChunks, { type: 'video/webm' });
            }
            
            // Convert blob to base64
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = function() {
                const base64data = reader.result;
                
                // Calculate stats
                const recordingDuration = (Date.now() - recordingStartTime) / 1000;
                const stats = {
                    total_frames: recordedChunks.length,
                    avg_fps: recordedChunks.length / recordingDuration,
                    drowsy_detections: statsData.drowsy_count,
                    yawn_detections: statsData.yawn_count,
                    head_movement_detections: statsData.head_count
                };
                
                // Send to server
                webcamStatus.textContent = "Saving recording...";
                fetch('/save_webcam_recording', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        detection_id: detectionId,
                        recording: base64data,
                        stats: stats
                    })
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Recording saved:', data);
                    
                    // Redirect to result page
                    window.location.href = `/view_result/${detectionId}`;
                })
                .catch(error => {
                    console.error('Error saving recording:', error);
                    webcamStatus.textContent = "Error saving recording. Please try again.";
                    webcamStatus.classList.add('text-danger');
                    
                    // Reset UI
                    startWebcamBtn.classList.remove('d-none');
                    stopWebcamBtn.classList.add('d-none');
                });
            };
        } catch (error) {
            console.error('Error processing recording:', error);
            webcamStatus.textContent = "Error processing recording. Please try again.";
            webcamStatus.classList.add('text-danger');
            
            // Reset UI
            startWebcamBtn.classList.remove('d-none');
            stopWebcamBtn.classList.add('d-none');
        }
        
        // Reset recording data
        recordedChunks = [];
    }
});
