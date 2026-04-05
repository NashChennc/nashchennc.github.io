/**
 * 解析 banner-title-layout.json，为内页横幅标题/副标题应用对齐、换行与列距。
 * 挂载点：含 [data-banner-layout-config] 的节点（与 [data-banner-layout-root] 可同一元素）。
 */
(function () {
  'use strict';

  var CLS_WRAP = 'is-banner-title-wrap';
  var CLS_STACK = 'is-banner-title-stack';

  function getMetricChildren(root, childSelector) {
    if (childSelector === ':scope > *') {
      return Array.prototype.slice.call(root.children);
    }
    return Array.prototype.slice.call(root.querySelectorAll(childSelector));
  }

  function measureTotalWidth(root, childSelector) {
    var children = getMetricChildren(root, childSelector);
    if (!children.length) {
      return { total: 0, gapPx: 0, count: 0 };
    }
    var total = 0;
    for (var i = 0; i < children.length; i++) {
      total += children[i].offsetWidth;
    }
    var cs = getComputedStyle(root);
    var gapStr = cs.gap || cs.columnGap || '0px';
    var gapPx = parseFloat(gapStr) || 0;
    var gaps = Math.max(0, children.length - 1) * gapPx;
    return { total: total + gaps, gapPx: gapPx, count: children.length };
  }

  function applyBase(root, entry) {
    if (entry.align) {
      root.setAttribute('data-banner-align', entry.align);
    }
    if (entry.columnGap) {
      root.style.setProperty('--banner-layout-gap', entry.columnGap);
    }
  }

  function applyWrapStack(root, entry, wrapMode) {
    var resp = entry.responsive || {};
    var ratio = typeof resp.overflowRatio === 'number' ? resp.overflowRatio : 1.02;
    var stackMax = typeof resp.stackMaxWidth === 'number' ? resp.stackMaxWidth : 0;
    var childSelector = entry.childSelector || '.banner-hero-title__col';

    root.classList.remove(CLS_WRAP, CLS_STACK);

    if (wrapMode === 'nowrap') {
      return;
    }
    if (wrapMode === 'wrap') {
      root.classList.add(CLS_WRAP);
      return;
    }
    if (wrapMode !== 'responsive') {
      return;
    }

    var cw = root.clientWidth;
    if (stackMax > 0 && cw > 0 && cw <= stackMax) {
      root.classList.add(CLS_STACK);
      return;
    }

    var m = measureTotalWidth(root, childSelector);
    if (m.count === 0) return;
    if (m.total > cw * ratio) {
      root.classList.add(CLS_WRAP);
    }
  }

  function bindEntry(root, entry) {
    var wrapMode = entry.wrapMode || 'responsive';
    applyBase(root, entry);

    if (wrapMode === 'responsive') {
      var ro = new ResizeObserver(function () {
        applyWrapStack(root, entry, wrapMode);
      });
      ro.observe(root);
      applyWrapStack(root, entry, wrapMode);
    } else {
      applyWrapStack(root, entry, wrapMode);
    }
  }

  function initFromConfig(config, mount) {
    if (!config || !config.entries || !config.entries.length) return;
    for (var i = 0; i < config.entries.length; i++) {
      var entry = config.entries[i];
      var sel = entry.selector || '.banner-hero-title--inner';
      var nodes = mount.querySelectorAll(sel);
      for (var j = 0; j < nodes.length; j++) {
        bindEntry(nodes[j], entry);
      }
    }
  }

  function boot() {
    var mount = document.querySelector('[data-banner-layout-root][data-banner-layout-config]');
    if (!mount) return;
    var url = mount.getAttribute('data-banner-layout-config');
    if (!url) return;

    fetch(url, { credentials: 'same-origin' })
      .then(function (r) {
        if (!r.ok) throw new Error('banner layout config ' + r.status);
        return r.json();
      })
      .then(function (cfg) {
        initFromConfig(cfg, mount);
      })
      .catch(function () {
        /* 保持纯 CSS 默认，不抛到控制台打扰读者 */
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
