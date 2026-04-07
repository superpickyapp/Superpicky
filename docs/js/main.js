/**
 * SuperPicky Website - Shared JavaScript
 * Language toggle + Lightbox functionality
 */

// === Language Toggle ===
function setLang(lang, evt) {
    document.documentElement.setAttribute('data-lang', lang);
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    if (evt && evt.target) {
        evt.target.classList.add('active');
    }
    localStorage.setItem('lang', lang);
}

// Load saved language on page load
(function initLang() {
    const savedLang = localStorage.getItem('lang') || 'cn';
    document.documentElement.setAttribute('data-lang', savedLang);
    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('.lang-btn').forEach(btn => {
            const isCN = btn.textContent.includes('中');
            if ((savedLang === 'cn' && isCN) || (savedLang === 'en' && !isCN)) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    });
})();

// === Lightbox ===
(function initLightbox() {
    document.addEventListener('DOMContentLoaded', () => {
        // Create lightbox elements
        const lightbox = document.createElement('div');
        lightbox.id = 'lightbox';
        lightbox.innerHTML = `
            <div class="lightbox-overlay"></div>
            <div class="lightbox-content">
                <img src="" alt="">
                <button class="lightbox-close">&times;</button>
            </div>
        `;
        document.body.appendChild(lightbox);

        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            #lightbox {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 10000;
            }
            #lightbox.active {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .lightbox-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.9);
                cursor: pointer;
            }
            .lightbox-content {
                position: relative;
                max-width: 95vw;
                max-height: 95vh;
                z-index: 1;
            }
            .lightbox-content img {
                max-width: 95vw;
                max-height: 95vh;
                object-fit: contain;
                border-radius: 8px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            }
            .lightbox-close {
                position: absolute;
                top: -40px;
                right: 0;
                background: none;
                border: none;
                color: white;
                font-size: 32px;
                cursor: pointer;
                opacity: 0.7;
                transition: opacity 0.2s;
            }
            .lightbox-close:hover {
                opacity: 1;
            }
            /* Make images clickable */
            .showcase-frame img,
            .hero-preview-frame img,
            .step-image img,
            .faq-image img {
                cursor: zoom-in;
                transition: transform 0.2s;
            }
            .showcase-frame img:hover,
            .hero-preview-frame img:hover,
            .step-image img:hover,
            .faq-image img:hover {
                transform: scale(1.02);
            }
        `;
        document.head.appendChild(style);

        const lightboxImg = lightbox.querySelector('img');
        const overlay = lightbox.querySelector('.lightbox-overlay');
        const closeBtn = lightbox.querySelector('.lightbox-close');

        // Open lightbox on image click
        document.addEventListener('click', (e) => {
            if (e.target.tagName === 'IMG' &&
                (e.target.closest('.showcase-frame') ||
                    e.target.closest('.hero-preview-frame') ||
                    e.target.closest('.step-image') ||
                    e.target.closest('.faq-image'))) {
                lightboxImg.src = e.target.src;
                lightbox.classList.add('active');
                document.body.style.overflow = 'hidden';
            }
        });

        // Close lightbox
        function closeLightbox() {
            lightbox.classList.remove('active');
            document.body.style.overflow = '';
        }

        overlay.addEventListener('click', closeLightbox);
        closeBtn.addEventListener('click', closeLightbox);
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeLightbox();
        });
    });
})();
