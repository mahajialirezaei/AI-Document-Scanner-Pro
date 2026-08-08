document.addEventListener('DOMContentLoaded', function() {
    const fileRaw = document.getElementById('fileRaw');
    const fileRef = document.getElementById('fileRef');
    const dropzoneRaw = document.getElementById('dropzoneRaw');
    const dropzoneRef = document.getElementById('dropzoneRef');
    const processBtn = document.getElementById('processBtn');
    
    let rawFile = null;
    let refFile = null;

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

    processBtn.addEventListener('click', () => {
        if(!rawFile) return;
        document.getElementById('results').classList.add('hidden');
        document.getElementById('metricsPanel').classList.add('hidden');
        document.getElementById('error').classList.add('hidden');
        document.getElementById('loading').classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', rawFile);
        if(refFile) formData.append('reference_file', refFile);
        
        formData.append('corner_method', document.getElementById('cornerMethod').value);
        formData.append('enhancement_method', document.getElementById('enhancementMethod').value);
        formData.append('apply_binarization', document.getElementById('applyBinarization').checked);

        fetch('/scan', { method: 'POST', body: formData })
        .then(res => res.json().then(data => res.ok ? data : Promise.reject(data)))
        .then(data => {
            document.getElementById('loading').classList.add('hidden');
            
            // Set Images
            document.getElementById('cornersImage').src = `data:image/jpeg;base64,${data.corners_image}`;
            document.getElementById('enhancedImage').src = `data:image/jpeg;base64,${data.enhanced_image}`;
            
            // Set Metrics
            const m = data.metrics;
            document.getElementById('m-psnr').innerText = m.psnr ? `${m.psnr.toFixed(2)} dB` : 'N/A (No GT)';
            document.getElementById('m-ssim').innerText = m.ssim ? m.ssim.toFixed(4) : 'N/A';
            document.getElementById('m-ocr-raw').innerText = m.ocr_raw !== undefined ? `${m.ocr_raw.toFixed(1)}%` : 'N/A';
            document.getElementById('m-ocr-enh').innerText = m.ocr_enhanced !== undefined ? `${m.ocr_enhanced.toFixed(1)}%` : 'N/A';
            document.getElementById('m-ocr-tgt').innerText = m.ocr_target !== undefined ? `${m.ocr_target.toFixed(1)}%` : '-';
            
            document.getElementById('metricsPanel').classList.remove('hidden');
            document.getElementById('results').classList.remove('hidden');
        })
        .catch(err => {
            document.getElementById('loading').classList.add('hidden');
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = err.detail || 'Processing failed.';
            errorDiv.classList.remove('hidden');
        });
    });
});