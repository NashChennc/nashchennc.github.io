(function () {
  'use strict';

  var toggle = document.getElementById('site-nav-toggle');
  var drawer = document.getElementById('site-category-drawer');
  if (!toggle || !drawer) return;

  function setExpanded(on) {
    toggle.setAttribute('aria-expanded', on ? 'true' : 'false');
  }

  function closeOnDesktop() {
    if (typeof mdui === 'undefined' || !mdui.breakpoint || !mdui.breakpoint().up('md')) return;
    if (drawer.open) drawer.open = false;
    setExpanded(false);
  }

  toggle.addEventListener('click', function () {
    drawer.open = !drawer.open;
    setExpanded(drawer.open);
  });

  drawer.addEventListener('open', function () {
    setExpanded(true);
  });

  drawer.addEventListener('close', function () {
    setExpanded(false);
  });

  drawer.addEventListener('closed', function () {
    setExpanded(false);
  });

  drawer.addEventListener('click', function (event) {
    var target = event.target;
    if (!target || typeof target.closest !== 'function') return;
    var link = target.closest('a[href], mdui-list-item[href]');
    if (!link || !drawer.contains(link)) return;
    drawer.open = false;
    setExpanded(false);
  });

  window.addEventListener(
    'resize',
    typeof mdui !== 'undefined' && typeof mdui.throttle === 'function'
      ? mdui.throttle(closeOnDesktop, 100)
      : closeOnDesktop
  );
  closeOnDesktop();
})();
