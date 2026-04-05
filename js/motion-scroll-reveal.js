/**
 * IntersectionObserver for .scroll-reveal → .is-visible (staggered).
 */
(function () {
  function init() {
    var observerOptions = {
      root: null,
      rootMargin: '0px',
      threshold: 0
    };

    var observer = new IntersectionObserver(function (entries, obs) {
      entries.forEach(function (entry, index) {
        if (entry.isIntersecting) {
          setTimeout(function () {
            entry.target.classList.add('is-visible');
          }, index * 100);
          obs.unobserve(entry.target);
        }
      });
    }, observerOptions);

    document.querySelectorAll('.scroll-reveal').forEach(function (el) {
      observer.observe(el);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
