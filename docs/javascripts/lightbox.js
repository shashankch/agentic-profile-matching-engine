/**
 * Yojaka AI — Documentation Portal Native Lightbox & Fullscreen Viewer
 * Zero-dependency, accessible, responsive image zoom & fullscreen utility.
 */

(function () {
  'use strict';

  let overlay = null;
  let currentImg = null;
  let isZoomed = false;
  let startX = 0;
  let startY = 0;
  let scrollLeft = 0;
  let scrollTop = 0;
  let isDragging = false;

  function createLightbox() {
    if (overlay) return overlay;

    overlay = document.createElement('div');
    overlay.id = 'mkdocs-lightbox';
    overlay.className = 'lightbox-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-hidden', 'true');

    overlay.innerHTML = `
      <div class="lightbox-backdrop"></div>
      <div class="lightbox-dialog">
        <div class="lightbox-toolbar">
          <div class="lightbox-caption" id="lightbox-caption-text">Architecture Diagram</div>
          <div class="lightbox-actions">
            <button type="button" class="lightbox-btn" id="lightbox-zoom-btn" title="Toggle Zoom (Z)" aria-label="Toggle Zoom">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                <line x1="11" y1="8" x2="11" y2="14"></line>
                <line x1="8" y1="11" x2="14" y2="11"></line>
              </svg>
              <span class="lightbox-btn-text">Zoom</span>
            </button>
            <button type="button" class="lightbox-btn" id="lightbox-fullscreen-btn" title="Toggle Fullscreen (F)" aria-label="Toggle Fullscreen">
              <svg id="lightbox-fullscreen-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
              </svg>
              <span class="lightbox-btn-text">Fullscreen</span>
            </button>
            <button type="button" class="lightbox-btn lightbox-close-btn" id="lightbox-close-btn" title="Close Viewer (Esc)" aria-label="Close">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
        </div>
        <div class="lightbox-stage" id="lightbox-stage">
          <img id="lightbox-display-img" class="lightbox-img" src="" alt="" draggable="false" />
        </div>
        <div class="lightbox-footer">
          <span class="lightbox-hint">Click image to toggle zoom &bull; ⛶ Fullscreen &bull; Press <kbd>Esc</kbd> or click backdrop to close</span>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);

    // Event listeners for overlay controls
    const backdrop = overlay.querySelector('.lightbox-backdrop');
    const closeBtn = overlay.querySelector('#lightbox-close-btn');
    const zoomBtn = overlay.querySelector('#lightbox-zoom-btn');
    const fsBtn = overlay.querySelector('#lightbox-fullscreen-btn');
    const displayImg = overlay.querySelector('#lightbox-display-img');
    const stage = overlay.querySelector('#lightbox-stage');

    backdrop.addEventListener('click', closeLightbox);
    closeBtn.addEventListener('click', closeLightbox);
    zoomBtn.addEventListener('click', toggleZoom);
    fsBtn.addEventListener('click', toggleFullscreen);

    displayImg.addEventListener('click', function (e) {
      e.stopPropagation();
      toggleZoom();
    });

    // Panning inside stage when zoomed
    stage.addEventListener('mousedown', function (e) {
      if (!isZoomed) return;
      isDragging = true;
      startX = e.pageX - stage.offsetLeft;
      startY = e.pageY - stage.offsetTop;
      scrollLeft = stage.scrollLeft;
      scrollTop = stage.scrollTop;
      stage.style.cursor = 'grabbing';
    });

    window.addEventListener('mouseup', function () {
      isDragging = false;
      if (stage && isZoomed) stage.style.cursor = 'grab';
    });

    stage.addEventListener('mousemove', function (e) {
      if (!isDragging || !isZoomed) return;
      e.preventDefault();
      const x = e.pageX - stage.offsetLeft;
      const y = e.pageY - stage.offsetTop;
      const walkX = (x - startX) * 1.5;
      const walkY = (y - startY) * 1.5;
      stage.scrollLeft = scrollLeft - walkX;
      stage.scrollTop = scrollTop - walkY;
    });

    window.addEventListener('keydown', function (e) {
      if (overlay.classList.contains('active')) {
        if (e.key === 'Escape') {
          e.preventDefault();
          closeLightbox();
        } else if (e.key === 'f' || e.key === 'F') {
          e.preventDefault();
          toggleFullscreen();
        } else if (e.key === 'z' || e.key === 'Z') {
          e.preventDefault();
          toggleZoom();
        }
      }
    });

    return overlay;
  }

  function openLightbox(img) {
    createLightbox();
    currentImg = img;
    isZoomed = false;

    const displayImg = overlay.querySelector('#lightbox-display-img');
    const captionText = overlay.querySelector('#lightbox-caption-text');
    const stage = overlay.querySelector('#lightbox-stage');

    displayImg.src = img.currentSrc || img.src;
    displayImg.alt = img.alt || 'Architecture Diagram';
    displayImg.classList.remove('zoomed');
    stage.classList.remove('is-zoomed');
    stage.scrollLeft = 0;
    stage.scrollTop = 0;

    captionText.textContent = img.alt || img.title || 'Architecture Specification & Flow Diagram';

    overlay.classList.add('active');
    overlay.setAttribute('aria-hidden', 'false');
    document.body.classList.add('lightbox-open');

    updateZoomButtonUI();
  }

  function closeLightbox() {
    if (!overlay) return;
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(function () {});
    }
    overlay.classList.remove('active');
    overlay.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('lightbox-open');
    isZoomed = false;
  }

  function toggleZoom() {
    if (!overlay) return;
    const displayImg = overlay.querySelector('#lightbox-display-img');
    const stage = overlay.querySelector('#lightbox-stage');

    isZoomed = !isZoomed;
    if (isZoomed) {
      displayImg.classList.add('zoomed');
      stage.classList.add('is-zoomed');
      stage.style.cursor = 'grab';
    } else {
      displayImg.classList.remove('zoomed');
      stage.classList.remove('is-zoomed');
      stage.style.cursor = 'default';
      stage.scrollLeft = 0;
      stage.scrollTop = 0;
    }
    updateZoomButtonUI();
  }

  function updateZoomButtonUI() {
    if (!overlay) return;
    const zoomBtn = overlay.querySelector('#lightbox-zoom-btn');
    const btnText = zoomBtn.querySelector('.lightbox-btn-text');
    if (btnText) {
      btnText.textContent = isZoomed ? 'Reset' : 'Zoom';
    }
  }

  function toggleFullscreen() {
    if (!overlay) return;
    if (!document.fullscreenElement) {
      if (overlay.requestFullscreen) {
        overlay.requestFullscreen();
      } else if (overlay.webkitRequestFullscreen) {
        overlay.webkitRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }

  function shouldExcludeImage(img) {
    const src = img.getAttribute('src') || '';
    if (
      src.includes('shields.io') ||
      src.includes('badge') ||
      src.includes('streamlit.io') ||
      src.includes('assets/icons/') ||
      img.closest('.hero-badges') ||
      img.classList.contains('no-lightbox') ||
      img.classList.contains('twemoji')
    ) {
      return true;
    }
    return false;
  }

  function initLightboxTriggers() {
    const images = document.querySelectorAll('.md-typeset img');
    images.forEach(function (img) {
      if (shouldExcludeImage(img)) return;

      if (!img.dataset.lightboxBound) {
        img.dataset.lightboxBound = 'true';
        img.classList.add('lightbox-trigger');
        img.setAttribute('title', 'Click to open fullscreen view & zoom');
        img.addEventListener('click', function (e) {
          // If image is inside a link to an external page, only open lightbox if it links to an image file or itself
          const link = img.closest('a');
          if (link) {
            const href = link.getAttribute('href') || '';
            if (/\.(png|jpg|jpeg|gif|svg|webp)(\?.*)?$/i.test(href) || href === '#' || href === '') {
              e.preventDefault();
              openLightbox(img);
            }
          } else {
            e.preventDefault();
            openLightbox(img);
          }
        });
      }
    });
  }

  // Bind on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initLightboxTriggers);
  } else {
    initLightboxTriggers();
  }

  // Support Material for MkDocs instant navigation
  if (typeof document$ !== 'undefined') {
    document$.subscribe(function () {
      initLightboxTriggers();
    });
  }
})();
