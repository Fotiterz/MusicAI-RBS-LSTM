document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('musicForm');
    const generateBtn = document.getElementById('generateBtn');
    const playerSection = document.getElementById('playerSection');
    const audioPlayer = document.getElementById('audioPlayer');
    const downloadBtn = document.getElementById('downloadBtn');
    const downloadType = document.getElementById('downloadType');

    if (!form || !generateBtn || !playerSection || !audioPlayer || !downloadBtn || !downloadType) {
        console.error('One or more required elements are missing in the DOM.');
        return;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        generateBtn.disabled = true;

        const mode = document.getElementById('mode').value;
        const tempo = document.getElementById('tempo').value;
        const key = document.getElementById('key').value;
        const length = document.getElementById('length').value;
        const instruments = Array.from(document.getElementById('instruments').selectedOptions).map(o => o.value);

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode, tempo, key, length, instruments })
            });

            if (!response.ok) {
                const errorText = await response.text();
                alert('Generation failed: ' + errorText);
                generateBtn.disabled = false;
                return;
            }

            const result = await response.json();

            if (result.mp3_url && result.midi_url) {
                playerSection.style.display = 'block';
                audioPlayer.src = result.mp3_url;
                audioPlayer.load();

                function updateDownloadLink() {
                    if (downloadType.value === 'mp3') {
                        downloadBtn.href = result.mp3_url;
                        downloadBtn.download = result.mp3_url.split('/').pop();
                    } else {
                        downloadBtn.href = result.midi_url;
                        downloadBtn.download = result.midi_url.split('/').pop();
                    }
                }

                updateDownloadLink();
                downloadType.addEventListener('change', updateDownloadLink);
                downloadBtn.style.display = 'inline-block';
            } else if (result.error) {
                alert('Error: ' + result.error);
                if (playerSection) playerSection.style.display = 'none';
                if (downloadBtn) downloadBtn.style.display = 'none';
            }
        } catch (err) {
            alert('Generation failed: ' + err.message);
            if (playerSection) playerSection.style.display = 'none';
            if (downloadBtn) downloadBtn.style.display = 'none';
        } finally {
            generateBtn.disabled = false;
        }
    });
});
