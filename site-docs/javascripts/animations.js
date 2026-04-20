/* ============================================================================
   ALMA landing page animations — lightweight, dependency-free.
   ---------------------------------------------------------------------------
   - Scroll-triggered reveals via IntersectionObserver (unobserve after reveal)
   - Animated counter for the benchmark value (respects prefers-reduced-motion)
   - Subtle parallax drift on the aurora orbs (requestAnimationFrame)

   Respects `prefers-reduced-motion: reduce` — all motion disabled if requested.
   Safe to run on every page; no-ops when landing-page elements are absent.
   ============================================================================ */

(function () {
  'use strict';

  const prefersReducedMotion = window.matchMedia(
    '(prefers-reduced-motion: reduce)'
  ).matches;

  // ---------- 1. Scroll reveals -------------------------------------------
  const reveals = document.querySelectorAll('.alma-reveal');
  if (reveals.length && !prefersReducedMotion && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('alma-revealed');
            observer.unobserve(entry.target);
          }
        });
      },
      { root: null, rootMargin: '0px 0px -8% 0px', threshold: 0.12 }
    );
    reveals.forEach((el) => io.observe(el));
  } else if (reveals.length) {
    // Fallback — reveal everything immediately
    reveals.forEach((el) => el.classList.add('alma-revealed'));
  }

  // ---------- 2. Counter animation ----------------------------------------
  const counters = document.querySelectorAll('[data-counter]');
  if (counters.length && !prefersReducedMotion && 'IntersectionObserver' in window) {
    const animateCounter = (el) => {
      const target = parseFloat(el.dataset.counter);
      const decimals = parseInt(el.dataset.decimals || '0', 10);
      if (isNaN(target)) return;

      const duration = 1400;
      const start = performance.now();

      const tick = (now) => {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        // easeOutCubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = (target * eased).toFixed(decimals);
        el.textContent = value;
        if (progress < 1) requestAnimationFrame(tick);
        else el.textContent = target.toFixed(decimals);
      };
      requestAnimationFrame(tick);
    };

    const counterIO = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateCounter(entry.target);
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.5 }
    );
    counters.forEach((el) => counterIO.observe(el));
  } else if (counters.length) {
    counters.forEach((el) => {
      const target = parseFloat(el.dataset.counter);
      const decimals = parseInt(el.dataset.decimals || '0', 10);
      if (!isNaN(target)) el.textContent = target.toFixed(decimals);
    });
  }

  // ---------- 3. Parallax drift on hero aurora orbs -----------------------
  const orbs = document.querySelectorAll('.alma-aurora-orb');
  if (orbs.length && !prefersReducedMotion) {
    let ticking = false;
    const applyParallax = () => {
      const y = window.scrollY;
      orbs.forEach((orb, i) => {
        const speed = (i + 1) * 0.08; // each orb moves at a different rate
        orb.style.setProperty('--parallax-y', `${y * speed}px`);
        orb.style.transform = `translate3d(0, ${y * speed}px, 0)`;
      });
      ticking = false;
    };
    window.addEventListener(
      'scroll',
      () => {
        if (!ticking) {
          requestAnimationFrame(applyParallax);
          ticking = true;
        }
      },
      { passive: true }
    );
  }
})();
