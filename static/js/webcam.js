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
            
            webcamStatus.textContent = "Starting webcam stream...";
            if (cameraIcon) {
                cameraIcon.style.display = 'none';
            }
            
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
                
                videoStream.src = `/video_feed/${detectionId}`;
                
                videoStream.classList.remove('d-none');
                // statsContainer.classList.remove('d-none');
                
                isDetecting = true;
                webcamStatus.textContent = "Detecting...";
                
                setupMediaRecording();
                
                stopWebcamBtn.classList.remove('d-none');
                startWebcamBtn.classList.add('d-none');
                
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
        recordingStartTime = Date.now();
        
        try {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 640;
            canvas.height = 480;
            
            const stream = canvas.captureStream(30); 
            
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp9',
                videoBitsPerSecond: 2500000 
            });
            
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            
            mediaRecorder.start(1000);
            
            const drawInterval = setInterval(() => {
                if (!isDetecting) {
                    clearInterval(drawInterval);
                    return;
                }
                
                if (videoStream.complete && videoStream.naturalHeight !== 0) {
                    ctx.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
                }
            }, 33); 
            
        } catch (error) {
            console.error('Error setting up media recording:', error);
            setupFrameRecording();
        }
    }
    
    function setupFrameRecording() {
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
            
            if (videoStream.complete && videoStream.naturalHeight !== 0) {
                ctx.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
                
                canvas.toBlob(blob => {
                    if (blob) {
                        recordedChunks.push(blob);
                    }
                }, 'image/jpeg', 0.85);
            }
        }, 100); 
    }
    
    function startStatsUpdate() {
        const statsInterval = setInterval(() => {
            if (!isDetecting) {
                clearInterval(statsInterval);
                return;
            }
            
            fetch(`/check_processing_status/${detectionId}`)
                .then(response => response.json())
                .then(data => {
                    statsData.drowsy_count = data.drowsy_count || 0;
                    statsData.yawn_count = data.yawn_count || 0;
                    statsData.head_count = data.head_movement_count || 0;
                    
                    document.getElementById('drowsyCount').textContent = statsData.drowsy_count;
                    document.getElementById('yawnCount').textContent = statsData.yawn_count;
                    document.getElementById('headCount').textContent = statsData.head_count;
                })
                .catch(error => {
                    console.error('Error fetching stats:', error);
                });
        }, 1000); 
    }
    
    function stopDetection() {
        isDetecting = false;
        webcamStatus.textContent = "Processing recording...";
        
        videoStream.classList.add('d-none');
        videoStream.src = '';
        if (cameraIcon) {
            cameraIcon.style.display = 'block';
        }
        
        fetch('/stop_webcam_detection', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                detection_id: detectionId
            })
        });
        
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            
            setTimeout(() => {
                processRecording();
            }, 500);
        } else {
            processRecording();
        }
    }
    
    function processRecording() {
        let blob;
        
        try {
            if (mediaRecorder) {
                blob = new Blob(recordedChunks, { type: 'video/webm' });
            } else {
                blob = new Blob(recordedChunks, { type: 'video/webm' });
            }
            
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = function() {
                const base64data = reader.result;
                
                const recordingDuration = (Date.now() - recordingStartTime) / 1000;
                const stats = {
                    total_frames: recordedChunks.length,
                    avg_fps: recordedChunks.length / recordingDuration,
                    drowsy_detections: statsData.drowsy_count,
                    yawn_detections: statsData.yawn_count,
                    head_movement_detections: statsData.head_count
                };
            
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
                    
                    window.location.href = `/view_result/${detectionId}`;
                })
                .catch(error => {
                    console.error('Error saving recording:', error);
                    webcamStatus.textContent = "Error saving recording. Please try again.";
                    webcamStatus.classList.add('text-danger');
                    
                    startWebcamBtn.classList.remove('d-none');
                    stopWebcamBtn.classList.add('d-none');
                });
            };
        } catch (error) {
            console.error('Error processing recording:', error);
            webcamStatus.textContent = "Error processing recording. Please try again.";
            webcamStatus.classList.add('text-danger');
            
            startWebcamBtn.classList.remove('d-none');
            stopWebcamBtn.classList.add('d-none');
        }
        
        recordedChunks = [];
    }
});
