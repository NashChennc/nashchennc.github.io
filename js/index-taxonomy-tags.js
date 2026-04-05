/**
 * 标签「显示更多 / 收起」：挤压动画作用在 overflow sleeve（含溢出 chip 与「收起」按钮），不滚动页面。
 */
(function () {
  var DURATION_MS = 340;
  var EASING = 'cubic-bezier(0.4, 0, 0.2, 1)';

  function reducedMotion() {
    try {
      return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (e) {
      return false;
    }
  }

  function initDetails(details) {
    if (reducedMotion()) return;

    var summary = details.querySelector('.index-taxonomy__tags-summary');
    var sleeve = details.querySelector('.index-taxonomy__tags-overflow-sleeve');
    var lessBtn = details.querySelector('.index-taxonomy__tags-less');
    if (!summary || !sleeve || !lessBtn) return;

    details.classList.add('index-taxonomy__tags-details--js');

    function clearAnimating() {
      details.dataset.tagSqueezeAnimating = '0';
    }

    function armFallback(done) {
      var t = window.setTimeout(done, DURATION_MS + 80);
      return function cancel() {
        window.clearTimeout(t);
      };
    }

    function runExpand() {
      details.open = true;
      sleeve.style.overflow = 'hidden';
      sleeve.style.maxHeight = '0';
      sleeve.style.opacity = '0';
      sleeve.style.transform = 'scaleY(0.94)';
      sleeve.style.transformOrigin = 'top center';
      void sleeve.offsetHeight;
      sleeve.style.transition =
        'max-height ' + DURATION_MS + 'ms ' + EASING + ', opacity ' + Math.round(DURATION_MS * 0.75) + 'ms ' + EASING + ', transform ' + DURATION_MS + 'ms ' + EASING;

      var targetH = sleeve.scrollHeight;
      var expandDone = false;
      var cancelFbExpand = armFallback(finishExpand);
      function finishExpand() {
        if (expandDone) return;
        expandDone = true;
        cancelFbExpand();
        sleeve.removeEventListener('transitionend', onExpandEnd);
        if (details.open) {
          sleeve.style.transition = '';
          sleeve.style.maxHeight = 'none';
          sleeve.style.opacity = '';
          sleeve.style.transform = '';
          sleeve.style.overflow = '';
          sleeve.style.transformOrigin = '';
        }
        clearAnimating();
      }
      function onExpandEnd(ev) {
        if (ev.target !== sleeve || ev.propertyName !== 'max-height') return;
        finishExpand();
      }
      sleeve.addEventListener('transitionend', onExpandEnd);

      requestAnimationFrame(function () {
        sleeve.style.maxHeight = targetH + 'px';
        sleeve.style.opacity = '1';
        sleeve.style.transform = 'scaleY(1)';
      });
    }

    function runCollapse() {
      sleeve.style.overflow = 'hidden';
      var h = sleeve.scrollHeight;
      sleeve.style.maxHeight = h + 'px';
      sleeve.style.opacity = '1';
      sleeve.style.transform = 'scaleY(1)';
      sleeve.style.transformOrigin = 'top center';
      void sleeve.offsetHeight;
      sleeve.style.transition =
        'max-height ' + DURATION_MS + 'ms ' + EASING + ', opacity ' + Math.round(DURATION_MS * 0.75) + 'ms ' + EASING + ', transform ' + DURATION_MS + 'ms ' + EASING;

      var collapseDone = false;
      var cancelFbCollapse = armFallback(finishCollapse);
      function finishCollapse() {
        if (collapseDone) return;
        collapseDone = true;
        cancelFbCollapse();
        sleeve.removeEventListener('transitionend', onCollapseEnd);
        details.open = false;
        sleeve.style.transition = '';
        sleeve.style.maxHeight = '';
        sleeve.style.opacity = '';
        sleeve.style.transform = '';
        sleeve.style.overflow = '';
        sleeve.style.transformOrigin = '';
        clearAnimating();
        try {
          summary.focus({ preventScroll: true });
        } catch (e1) {
          try {
            summary.focus();
          } catch (e2) {}
        }
      }
      function onCollapseEnd(ev) {
        if (ev.target !== sleeve || ev.propertyName !== 'max-height') return;
        finishCollapse();
      }
      sleeve.addEventListener('transitionend', onCollapseEnd);

      sleeve.style.maxHeight = '0';
      sleeve.style.opacity = '0';
      sleeve.style.transform = 'scaleY(0.94)';
    }

    summary.addEventListener('click', function (e) {
      if (details.dataset.tagSqueezeAnimating === '1') {
        e.preventDefault();
        return;
      }
      if (details.open) return;
      e.preventDefault();
      details.dataset.tagSqueezeAnimating = '1';
      runExpand();
    });

    lessBtn.addEventListener('click', function (e) {
      if (details.dataset.tagSqueezeAnimating === '1') {
        e.preventDefault();
        return;
      }
      if (!details.open) return;
      e.preventDefault();
      details.dataset.tagSqueezeAnimating = '1';
      runCollapse();
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.index-taxonomy__tags-details').forEach(initDetails);
  });
})();
