document.addEventListener('DOMContentLoaded', function() {
    let mode = 'auto'; // 'auto' or 'interactive'
    
    const fileRaw = document.getElementById('fileRaw');
    const fileRef = document.getElementById('fileRef');
    const dropzoneRaw = document.getElementById('dropzoneRaw');
    const dropzoneRef = document.getElementById('dropzoneRef');
    const processBtn = document.getElementById('processBtn');
    
    // Tabs
    const tabAuto = document.getElementById('tabAuto');
    const tabInteractive = document.getElementById('tabInteractive');
    const autoView = document.getElementById('autoView');
    const interactiveView = document.getElementById('interactiveView');
    const dropzoneRefContainer = document.getElementById('dropzoneRef');

    let rawFile = null;
    let refFile = null;
    
    // Canvas variables
    const canvas = document.getElementById('editorCanvas');
    const ctx = canvas.getContext('2d');
    let imgObj = new Image();
    let cornersNorm = []; // Normalized corners [0..1]
    let draggingPoint = -1;

    // Tab Switching Logic
    tabAuto.addEventListener('click', () => {
        mode = 'auto';
        tabAuto.classList.add('active');
        tabInteractive.classList.remove('active');
        autoView.classList.remove('hidden');
        interactiveView.classList.add('hidden');
        dropzoneRefContainer.classList.remove('hidden');
        processBtn.innerText = "Start Auto Processing";
    });

    tabInteractive.addEventListener('click', () => {
        mode = 'interactive';
        tabInteractive.classList.add('active');
        tabAuto.classList.remove('active');
        interactiveView.classList.remove('hidden');
        autoView.classList.add('hidden');
        dropzoneRefContainer.classList.add('hidden');
        processBtn.innerText = "Detect Corners for Editor";
    });

    dropzoneRaw.addEventListener('click', () => fileRaw.click());
    dropzoneRef.addEventListener('click', () => fileRef.click());

    fileRaw.addEventListener('change', (e) => {
        if(e.target.files.length) {
            rawFile = e.target.files[0];
            dropzoneRaw.querySelector('.dropzone-text').innerHTML = `✅ ${rawFile.name}`;
            processBtn.disabled = false;
        }
    });

    fileRef.addEventListener('change', (e) => {
        if(e.target.files.length) {
            refFile = e.target.files[0];
            dropzoneRef.querySelector('.dropzone-text').innerHTML = `✅ ${refFile.name}`;
        }
    });

    // Main Process Button Logic
    processBtn.addEventListener('click', () => {
        if(!rawFile) return;
        
        document.getElementById('error').classList.add('hidden');
        document.getElementById('loading').classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', rawFile);
        formData.append('corner_method', document.getElementById('cornerMethod').value);
        
        if (mode === 'auto') {
            document.getElementById('results').classList.add('hidden');
            document.getElementById('metricsPanel').classList.add('hidden');
            
            if(refFile) formData.append('reference_file', refFile);
            formData.append('enhancement_method', document.getElementById('enhancementMethod').value);
            formData.append('apply_binarization', document.getElementById('applyBinarization').checked);
            formData.append('apply_ink_boost', document.getElementById('applyInkBoost').checked);

            fetch('/scan', { method: 'POST', body: formData })
            .then(res => {
                if (res.headers.get("content-type") === "application/pdf") {
                    return res.blob().then(blob => ({ isPdf: true, blob: blob }));
                }
                return res.json().then(data => res.ok ? data : Promise.reject(data));
            })
            .then(data => {
                document.getElementById('loading').classList.add('hidden');
                if (data.isPdf) {
                    const url = window.URL.createObjectURL(data.blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = "enhanced_scan.pdf";
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    alert("PDF Processed and Downloaded Successfully!");
                    return;
                }
                
                document.getElementById('cornersImage').src = `data:image/jpeg;base64,${data.corners_image}`;
                document.getElementById('enhancedImage').src = `data:image/jpeg;base64,${data.enhanced_image}`;
                
                const m = data.metrics;
                document.getElementById('m-psnr').innerText = m.psnr ? `${m.psnr.toFixed(2)} dB` : 'N/A (No GT)';
                document.getElementById('m-ssim').innerText = m.ssim ? m.ssim.toFixed(4) : 'N/A';
                document.getElementById('m-ocr-raw').innerText = m.ocr_raw !== undefined ? `${m.ocr_raw.toFixed(1)}%` : 'N/A';
                document.getElementById('m-ocr-enh').innerText = m.ocr_enhanced !== undefined ? `${m.ocr_enhanced.toFixed(1)}%` : 'N/A';
                document.getElementById('m-ocr-tgt').innerText = m.ocr_target !== undefined ? `${m.ocr_target.toFixed(1)}%` : '-';
                
                document.getElementById('metricsPanel').classList.remove('hidden');
                document.getElementById('results').classList.remove('hidden');
            })
            .catch(handleError);

        } else if (mode === 'interactive') {
            if (rawFile.type === "application/pdf") {
                handleError({detail: "Interactive mode only supports images."});
                return;
            }
            // Load image to canvas
            const url = URL.createObjectURL(rawFile);
            imgObj.onload = () => {
                canvas.width = imgObj.width > 800 ? 800 : imgObj.width;
                canvas.height = (imgObj.height / imgObj.width) * canvas.width;
                
                // Fetch initial corners from API
                fetch('/interactive-detect', { method: 'POST', body: formData })
                .then(res => res.json().then(data => res.ok ? data : Promise.reject(data)))
                .then(data => {
                    document.getElementById('loading').classList.add('hidden');
                    cornersNorm = data.corners;
                    drawCanvas();
                })
                .catch(handleError);
            };
            imgObj.src = url;
        }
    });

    // Canvas Logic
    function drawCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(imgObj, 0, 0, canvas.width, canvas.height);
        
        if (cornersNorm.length !== 4) return;

        ctx.strokeStyle = '#2ecc71';
        ctx.lineWidth = 3;
        ctx.beginPath();
        cornersNorm.forEach((pt, i) => {
            const x = pt[0] * canvas.width;
            const y = pt[1] * canvas.height;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();

        ctx.fillStyle = '#3498db';
        cornersNorm.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt[0] * canvas.width, pt[1] * canvas.height, 8, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        });
    }

    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        for (let i = 0; i < cornersNorm.length; i++) {
            const ptX = cornersNorm[i][0] * canvas.width;
            const ptY = cornersNorm[i][1] * canvas.height;
            
            if (Math.hypot(mouseX - ptX, mouseY - ptY) < 25) {
                draggingPoint = i;
                break;
            }
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (draggingPoint === -1) return;
        const rect = canvas.getBoundingClientRect();
        
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;

        cornersNorm[draggingPoint][0] = Math.max(0, Math.min(1, x));
        cornersNorm[draggingPoint][1] = Math.max(0, Math.min(1, y));
        drawCanvas();
    });

    canvas.addEventListener('mouseup', () => draggingPoint = -1);
    canvas.addEventListener('mouseleave', () => draggingPoint = -1);

    // Interactive Enhance Submit
    document.getElementById('enhanceCustomBtn').addEventListener('click', () => {
        document.getElementById('loading').classList.remove('hidden');
        document.getElementById('interactiveResultCard').classList.add('hidden');

        const formData = new FormData();
        formData.append('file', rawFile);
        formData.append('corners', JSON.stringify(cornersNorm));
        formData.append('enhancement_method', document.getElementById('enhancementMethod').value);
        formData.append('apply_binarization', document.getElementById('applyBinarization').checked);
        formData.append('apply_ink_boost', document.getElementById('applyInkBoost').checked);

        fetch('/interactive-enhance', { method: 'POST', body: formData })
        .then(res => res.json().then(data => res.ok ? data : Promise.reject(data)))
        .then(data => {
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('interactiveEnhancedImage').src = `data:image/jpeg;base64,${data.enhanced_image}`;
            document.getElementById('interactiveResultCard').classList.remove('hidden');
        })
        .catch(handleError);
    });

    function handleError(err) {
        document.getElementById('loading').classList.add('hidden');
        const errorDiv = document.getElementById('error');
        errorDiv.textContent = err.detail || 'Processing failed.';
        errorDiv.classList.remove('hidden');
    }
});