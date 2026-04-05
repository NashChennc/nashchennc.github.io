(function () {
  'use strict';

  var dialog = document.getElementById('search-palette');
  var toggle = document.getElementById('site-search-toggle');
  var field = document.getElementById('palette-input');
  var resultsList = document.getElementById('palette-results');
  var emptyState = document.getElementById('palette-empty-state');
  if (!dialog || !toggle || !field || !resultsList || !emptyState) return;

  var emptyHintEl = emptyState.querySelector('.palette-empty-state__text');
  var searchUrl = dialog.getAttribute('data-search-url') || '';
  var noResultsText = dialog.getAttribute('data-search-no-results') || 'No results';
  var emptyHintText = dialog.getAttribute('data-search-empty-hint') || '';

  var cache = null;
  var cachePromise = null;
  var selectedIndex = -1;
  var resultItems = [];

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function highlightText(text, q) {
    if (!q) return escapeHtml(text);
    var ql = q.toLowerCase();
    var lower = String(text).toLowerCase();
    var idx = lower.indexOf(ql);
    if (idx === -1) return escapeHtml(text);
    var parts = [];
    var start = 0;
    var escapedQ = escapeHtml(q);
    while (idx !== -1) {
      parts.push(escapeHtml(String(text).slice(start, idx)));
      parts.push('<mark class="site-search-mark">' + escapedQ + '</mark>');
      start = idx + q.length;
      idx = lower.indexOf(ql, start);
    }
    parts.push(escapeHtml(String(text).slice(start)));
    return parts.join('');
  }

  function stripHtml(html) {
    return String(html)
      .replace(/<script[\s\S]*?<\/script>/gi, ' ')
      .replace(/<style[\s\S]*?<\/style>/gi, ' ')
      .replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function snippetFromContent(content, q, maxLen) {
    maxLen = maxLen || 120;
    var plain = stripHtml(content || '');
    if (!q) return plain.slice(0, maxLen) + (plain.length > maxLen ? '…' : '');
    var lower = plain.toLowerCase();
    var ql = q.toLowerCase();
    var i = lower.indexOf(ql);
    if (i === -1) return plain.slice(0, maxLen) + (plain.length > maxLen ? '…' : '');
    var pad = 40;
    var start = Math.max(0, i - pad);
    var end = Math.min(plain.length, i + q.length + pad);
    var seg = plain.slice(start, end);
    var prefix = start > 0 ? '…' : '';
    var suffix = end < plain.length ? '…' : '';
    return prefix + seg + suffix;
  }

  function loadIndex() {
    if (cache) return Promise.resolve(cache);
    if (cachePromise) return cachePromise;
    cachePromise = fetch(searchUrl, { credentials: 'same-origin' })
      .then(function (res) {
        if (!res.ok) throw new Error('search fetch failed');
        return res.json();
      })
      .then(function (data) {
        cache = Array.isArray(data) ? data : [];
        return cache;
      })
      .catch(function () {
        cache = [];
        return cache;
      });
    return cachePromise;
  }

  function isDialogOpen() {
    return dialog.open === true;
  }

  function setToggleExpanded(on) {
    toggle.setAttribute('aria-expanded', on ? 'true' : 'false');
  }

  var SEL_CLASS = 'palette-row--sel';

  function clearSelection() {
    for (var i = 0; i < resultItems.length; i++) {
      resultItems[i].classList.remove(SEL_CLASS);
    }
    selectedIndex = -1;
  }

  function applySelection(idx) {
    if (!resultItems.length) return;
    if (idx < 0) idx = resultItems.length - 1;
    if (idx >= resultItems.length) idx = 0;
    for (var i = 0; i < resultItems.length; i++) {
      resultItems[i].classList.toggle(SEL_CLASS, i === idx);
    }
    selectedIndex = idx;
    resultItems[idx].scrollIntoView({ block: 'nearest' });
  }

  function renderResults(matches, queryRaw) {
    var query = (queryRaw || '').trim();
    resultsList.innerHTML = '';
    resultItems = [];
    clearSelection();

    if (!matches.length) {
      resultsList.hidden = true;
      emptyState.hidden = false;
      if (emptyHintEl) emptyHintEl.textContent = noResultsText;
      return;
    }

    resultsList.hidden = false;
    emptyState.hidden = true;

    for (var j = 0; j < matches.length; j++) {
      (function (index, m) {
        var item = document.createElement('mdui-list-item');
        item.setAttribute('href', m.url);
        item.setAttribute('icon', 'article');
        item.setAttribute('rounded', '');
        item.setAttribute('headline-line', '1');
        item.setAttribute('description-line', '2');

        var titleSpan = document.createElement('span');
        titleSpan.innerHTML = highlightText(m.title || '(untitled)', query);
        item.appendChild(titleSpan);

        var sn = snippetFromContent(m.content, query);
        if (sn) {
          var desc = document.createElement('span');
          desc.slot = 'description';
          desc.innerHTML = highlightText(sn, query);
          item.appendChild(desc);
        }

        item.addEventListener('mouseenter', function () {
          applySelection(index);
        });

        resultsList.appendChild(item);
        resultItems.push(item);
      })(j, matches[j]);
    }
  }

  function runSearch(q) {
    var query = (q || '').trim().toLowerCase();
    resultsList.innerHTML = '';
    resultItems = [];
    clearSelection();

    if (!query) {
      resultsList.hidden = true;
      emptyState.hidden = false;
      if (emptyHintEl) emptyHintEl.textContent = emptyHintText;
      return;
    }

    loadIndex().then(function (items) {
      var matches = [];
      for (var i = 0; i < items.length; i++) {
        var it = items[i] || {};
        var title = it.title != null ? String(it.title) : '';
        var content = it.content != null ? String(it.content) : '';
        var titleHit = title.toLowerCase().indexOf(query) !== -1;
        var bodyHit = content.toLowerCase().indexOf(query) !== -1;
        if (titleHit || bodyHit) {
          matches.push({
            title: title,
            url: it.url || '#',
            content: content,
            order: titleHit ? 0 : 1
          });
        }
      }
      matches.sort(function (a, b) {
        if (a.order !== b.order) return a.order - b.order;
        return 0;
      });

      renderResults(matches, q.trim());
    });
  }

  var throttledRun =
    typeof mdui !== 'undefined' && typeof mdui.throttle === 'function'
      ? mdui.throttle(runSearch, 200)
      : runSearch;

  function openDialog() {
    dialog.open = true;
    setToggleExpanded(true);
    setTimeout(function () {
      field.focus();
    }, 100);
  }

  function closeDialog() {
    dialog.open = false;
  }

  function isPathInsideDialog(path) {
    for (var i = 0; i < path.length; i++) {
      if (path[i] === dialog) return true;
    }
    return false;
  }

  function isPathSearchToggle(path) {
    for (var i = 0; i < path.length; i++) {
      var n = path[i];
      if (n && n.nodeType === 1 && n.id === 'site-search-toggle') return true;
    }
    return false;
  }

  function onDocumentPointerDownClose(e) {
    if (!isDialogOpen()) return;
    var path = typeof e.composedPath === 'function' ? e.composedPath() : [];
    if (!path.length) return;
    if (isPathInsideDialog(path)) return;
    if (isPathSearchToggle(path)) return;
    closeDialog();
  }

  function resetPaletteUi() {
    field.value = '';
    resultsList.innerHTML = '';
    resultItems = [];
    clearSelection();
    resultsList.hidden = true;
    emptyState.hidden = false;
    if (emptyHintEl) emptyHintEl.textContent = emptyHintText;
  }

  toggle.addEventListener('click', function () {
    if (isDialogOpen()) {
      closeDialog();
      return;
    }
    openDialog();
  });

  dialog.addEventListener('opened', function () {
    document.documentElement.classList.remove('mdui-lock-screen');
    document.documentElement.style.width = '';
    try {
      if (dialog.modalHelper && typeof dialog.modalHelper.deactivate === 'function') {
        dialog.modalHelper.deactivate();
      }
    } catch (err) {
      /* ignore */
    }
  });

  document.addEventListener('pointerdown', onDocumentPointerDownClose, true);

  dialog.addEventListener('closed', function () {
    setToggleExpanded(false);
    resetPaletteUi();
    toggle.focus();
  });

  document.addEventListener('keydown', function (e) {
    if (!(e.metaKey || e.ctrlKey) || (e.key !== 'k' && e.key !== 'K')) return;

    if (isDialogOpen()) {
      e.preventDefault();
      closeDialog();
      return;
    }

    var t = e.target;
    if (t && t.isContentEditable) return;
    var tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    e.preventDefault();
    openDialog();
  });

  field.addEventListener('input', function () {
    var v = field.value || '';
    if (!v.trim()) {
      resultsList.innerHTML = '';
      resultItems = [];
      clearSelection();
      resultsList.hidden = true;
      emptyState.hidden = false;
      if (emptyHintEl) emptyHintEl.textContent = emptyHintText;
      return;
    }
    throttledRun(v);
  });

  field.addEventListener('keydown', function (e) {
    if (!isDialogOpen()) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!resultItems.length) return;
      var next = selectedIndex < 0 ? 0 : (selectedIndex + 1) % resultItems.length;
      applySelection(next);
      return;
    }

    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (!resultItems.length) return;
      var prev;
      if (selectedIndex < 0) {
        prev = resultItems.length - 1;
      } else {
        prev = selectedIndex - 1;
        if (prev < 0) prev = resultItems.length - 1;
      }
      applySelection(prev);
      return;
    }

    if (e.key === 'Enter') {
      if (!resultItems.length) return;
      var idx = selectedIndex < 0 ? 0 : selectedIndex;
      var item = resultItems[idx];
      var href = item && item.getAttribute('href');
      if (!href || href === '#') return;
      e.preventDefault();
      window.location.href = href;
      closeDialog();
    }
  });

})();
