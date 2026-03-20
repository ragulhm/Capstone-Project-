document.addEventListener('DOMContentLoaded', () => {
    const textArea = document.getElementById('id_text');
    if (!textArea) {
        return;
    }

    textArea.addEventListener('input', () => {
        textArea.classList.remove('is-invalid');
    });
});
