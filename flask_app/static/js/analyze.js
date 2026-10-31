function selectSample(path) {
    document.getElementById('videoPath').value = path;
}

async function uploadVideo() {
    const fileInput = document.getElementById('videoFile');
    const statusDiv = document.getElementById('uploadStatus');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    statusDiv.style.display = 'block';
    statusDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.valid) {
            document.getElementById('videoPath').value = result.path;
            statusDiv.innerHTML = '<i class="fas fa-check-circle"></i> Uploaded: ' + result.filename;
            statusDiv.style.color = 'var(--accent-green)';
        } else {
            statusDiv.innerHTML = '<i class="fas fa-exclamation-circle"></i> ' + result.error;
            statusDiv.style.color = 'var(--accent-red)';
        }
    } catch (error) {
        statusDiv.innerHTML = '<i class="fas fa-exclamation-circle"></i> Upload failed';
        statusDiv.style.color = 'var(--accent-red)';
    }
}

async function analyzeVideo() {
    const videoPath = document.getElementById('videoPath').value;
    const modelName = document.getElementById('modelSelect').value;
    const resultCard = document.getElementById('resultCard');
    
    if (!videoPath) {
        alert('Please enter a video path or upload a video');
        return;
    }
    
    const btn = document.querySelector('.btn-analyze');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    btn.disabled = true;
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                video_path: videoPath,
                model_name: modelName
            })
        });
        
        const result = await response.json();
        
        if (result.valid) {
            document.getElementById('predClass').textContent = result.predicted_class.toUpperCase();
            
            const probsList = document.getElementById('probsList');
            probsList.innerHTML = '';
            for (const [cls, prob] of Object.entries(result.probabilities)) {
                const probPercent = (prob * 100).toFixed(1);
                probsList.innerHTML += `
                    <div class="prob-item">
                        <span class="prob-label">${cls}</span>
                        <div class="prob-bar-bg">
                            <div class="prob-bar" style="width: ${probPercent}%"></div>
                        </div>
                        <span class="prob-value">${probPercent}%</span>
                    </div>
                `;
            }
            
            const metaGrid = document.getElementById('metaGrid');
            metaGrid.innerHTML = `
                <div class="meta-item">
                    <span class="meta-label">Filename</span>
                    <span class="meta-value">${result.metadata.filename}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Resolution</span>
                    <span class="meta-value">${result.metadata.resolution}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">FPS</span>
                    <span class="meta-value">${result.metadata.fps.toFixed(2)}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Duration</span>
                    <span class="meta-value">${result.metadata.duration_sec.toFixed(2)}s</span>
                </div>
            `;
            
            resultCard.style.display = 'block';
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}
