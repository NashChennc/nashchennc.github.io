(function () {
  'use strict';

  // 仅在 mdui 的 tooltip 组件可用时启用（mdui.global.js 在 <head> 同步加载）
  if (typeof customElements === 'undefined' || !customElements.get('mdui-tooltip')) return;

  // 预览范围：post / page / index 三个页面的正文链接
  var SCOPE_SELECTOR = [
    '.post-page-card__body .mdui-prose a[href]',
    '.mdui-prose.page-page__prose a[href]',
    '.notes-subtitle a[href]',
    '.post-backlinks__list a[href]'
  ].join(', ');

  var EXCERPT_LEN = 120;
  var URL_LEN = 80;

  var cache = null;
  var cachePromise = null;

  function truncate(text, maxLen) {
    var s = String(text || '');
    return s.length > maxLen ? s.slice(0, maxLen) + '…' : s;
  }

  // 归一化 pathname：去 /index.html、去尾斜杠、解码，用于 link-previews.json 条目与链接的匹配
  // pretty_urls(trailing_index/trailing_html) 下条目 url 可能是 /about/index.html，需与链接 /about/ 统一
  function normalizePath(path) {
    var s = String(path || '');
    s = s.replace(/(^|\/)index\.html?$/i, '$1');
    s = s.replace(/\/+$/, '') || '/';
    try {
      return decodeURIComponent(s);
    } catch (e) {
      return s;
    }
  }

  // 懒加载 link-previews.json（生成期 title+description 索引），缓存结果
  function loadPreviewIndex() {
    if (cache) return Promise.resolve(cache);
    if (cachePromise) return cachePromise;
    cachePromise = fetch('/link-previews.json', { credentials: 'same-origin' })
      .then(function (res) {
        if (!res.ok) throw new Error('link-previews.json fetch failed');
        return res.json();
      })
      .then(function (data) {
        cache = new Map();
        (Array.isArray(data) ? data : []).forEach(function (entry) {
          if (!entry || !entry.url) return;
          cache.set(entry.url, entry);              // 原始 url 精确匹配
          cache.set(normalizePath(entry.url), entry); // 归一化 pathname 匹配
        });
        return cache;
      })
      .catch(function () {
        cache = new Map();
        return cache;
      });
    return cachePromise;
  }

  function isInternal(href) {
    try {
      var u = new URL(href, location.href);
      return (u.protocol === 'http:' || u.protocol === 'https:') &&
        u.hostname === location.hostname;
    } catch (e) {
      return false;
    }
  }

  function isSkippableHref(href) {
    var t = String(href || '').trim();
    if (!t || t === '#' || t.charAt(0) === '#') return true;
    var low = t.toLowerCase();
    return low.indexOf('mailto:') === 0 ||
      low.indexOf('tel:') === 0 ||
      low.indexOf('javascript:') === 0 ||
      low.indexOf('data:') === 0 ||
      low.indexOf('blob:') === 0 ||
      low.indexOf('file:') === 0;
  }

  function isSkippableAnchor(a) {
    var href = a.getAttribute('href');
    if (!href || isSkippableHref(href)) return true;
    if (a.hasAttribute('data-link-preview')) return true;
    if (a.closest('pre, code')) return true;
    // 纯图片链接（无文本）不预览，避免干扰图片 / 画廊
    var text = (a.textContent || '').trim();
    if (!text && a.querySelector('img')) return true;
    return false;
  }

  function createTooltip(a) {
    var tooltip = document.createElement('mdui-tooltip');
    tooltip.variant = 'rich';
    tooltip.trigger = 'hover';
    tooltip.placement = 'auto';
    tooltip.openDelay = 300;
    tooltip.closeDelay = 150;
    tooltip.disabled = true; // 内容就绪前不弹出

    var parent = a.parentNode;
    parent.insertBefore(tooltip, a);
    tooltip.appendChild(a);
    a.setAttribute('data-link-preview', '1');
    // 移除原生 title，避免与 mdui tooltip 叠加显示
    a.removeAttribute('title');
    return tooltip;
  }

  function fillExternal(tooltip, href) {
    var host = '';
    try {
      host = new URL(href, location.href).hostname;
    } catch (e) { /* ignore */ }
    tooltip.headline = host || href;
    tooltip.content = truncate(href, URL_LEN);
    tooltip.disabled = false;
  }

  function safeDecode(str) {
    try {
      return decodeURIComponent(str);
    } catch (e) {
      return str;
    }
  }

  function lookupEntry(href) {
    if (!cache) return null;
    try {
      var u = new URL(href, location.href);
      return cache.get(u.pathname) || cache.get(normalizePath(u.pathname));
    } catch (e) {
      return null;
    }
  }

  // 识别 /categories/<name>/ 与 /tags/<name>/ 站内归档链接
  function parseTaxonomy(href) {
    var u;
    try {
      u = new URL(href, location.href);
    } catch (e) {
      return null;
    }
    var m = u.pathname.match(/^\/(categories|tags)\/(.+?)\/?$/);
    if (!m) return null;
    return { type: m[1], name: safeDecode(m[2]) };
  }

  function fillInternal(tooltip, href, linkText) {
    // 1. 站内 post / page：标题 + 摘要（description）
    var entry = lookupEntry(href);
    if (entry && entry.title) {
      tooltip.headline = entry.title;
      tooltip.content = entry.description || '';
      tooltip.disabled = false;
      return;
    }
    // 2. 站内 categories / tags 归档：名称 + 类型
    var tax = parseTaxonomy(href);
    if (tax) {
      tooltip.headline = tax.name;
      tooltip.content = tax.type === 'tags' ? '标签' : '分类';
      tooltip.disabled = false;
      return;
    }
    // 3. 其它站内链接（如 PDF 附件等未收录资源）：链接文本 + 路径，而非站点域名
    var u = null;
    try {
      u = new URL(href, location.href);
    } catch (e) { /* ignore */ }
    var text = String(linkText || '').trim();
    tooltip.headline = text
      ? truncate(text, EXCERPT_LEN)
      : (u ? safeDecode(u.pathname) : href);
    tooltip.content = u
      ? truncate(safeDecode(u.pathname), URL_LEN)
      : truncate(href, URL_LEN);
    tooltip.disabled = false;
  }

  function init() {
    var anchors = document.querySelectorAll(SCOPE_SELECTOR);
    var internal = [];

    Array.prototype.forEach.call(anchors, function (a) {
      if (isSkippableAnchor(a)) return;
      var href = a.getAttribute('href');
      var tooltip = createTooltip(a);
      if (isInternal(href)) {
        internal.push({ tooltip: tooltip, href: href, text: a.textContent });
      } else {
        fillExternal(tooltip, href);
      }
    });

    if (internal.length) {
      loadPreviewIndex().then(function () {
        internal.forEach(function (item) {
          fillInternal(item.tooltip, item.href, item.text);
        });
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
