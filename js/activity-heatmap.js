(function () {
  'use strict';

  var DAY_MS = 24 * 60 * 60 * 1000;

  function pad2(n) {
    return n < 10 ? '0' + n : String(n);
  }

  function dateFromKey(key) {
    var parts = String(key || '').split('-').map(function (part) {
      return parseInt(part, 10);
    });
    if (parts.length !== 3 || parts.some(function (part) { return !Number.isFinite(part); })) return null;
    return new Date(parts[0], parts[1] - 1, parts[2]);
  }

  function keyFromDate(date) {
    return date.getFullYear() + '-' + pad2(date.getMonth() + 1) + '-' + pad2(date.getDate());
  }

  function addDays(date, count) {
    var d = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    d.setDate(d.getDate() + count);
    return d;
  }

  function startOfWeek(date) {
    return addDays(date, -date.getDay());
  }

  function endOfWeek(date) {
    return addDays(date, 6 - date.getDay());
  }

  function safeLocale() {
    var lang = document.documentElement.getAttribute('lang') || navigator.language || 'en-US';
    if (lang === 'cn') return 'zh-CN';
    return lang;
  }

  function formatMonth(date) {
    try {
      return new Intl.DateTimeFormat(safeLocale(), { month: 'short' }).format(date);
    } catch (e) {
      return String(date.getMonth() + 1);
    }
  }

  function formatWeekday(date) {
    try {
      return new Intl.DateTimeFormat(safeLocale(), { weekday: 'short' }).format(date);
    } catch (e) {
      return '';
    }
  }

  function formatReadableDate(key) {
    var date = dateFromKey(key);
    if (!date) return key;
    try {
      return new Intl.DateTimeFormat(safeLocale(), {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      }).format(date);
    } catch (e) {
      return key;
    }
  }

  function getLabel(root, name, fallback) {
    return root.getAttribute('data-label-' + name) || fallback;
  }

  function countLabel(root, count) {
    var singular = getLabel(root, 'post', 'post');
    var plural = getLabel(root, 'posts', 'posts');
    return count + ' ' + (count === 1 ? singular : plural);
  }

  function groupByDate(posts) {
    var map = new Map();
    posts.forEach(function (post) {
      if (!map.has(post.date)) map.set(post.date, []);
      map.get(post.date).push(post);
    });
    map.forEach(function (items) {
      items.sort(function (a, b) {
        return String(a.title).localeCompare(String(b.title), undefined, { sensitivity: 'base' });
      });
    });
    return map;
  }

  function levelFor(count, maxCount) {
    if (!count) return 0;
    if (!maxCount) return 0;
    return Math.max(1, Math.ceil((count / maxCount) * 4));
  }

  function initWeekdays(root) {
    var container = root.querySelector('[data-activity-weekdays]');
    if (!container || container.childElementCount) return;
    var sunday = new Date(2024, 0, 7);
    for (var i = 0; i < 7; i++) {
      var item = document.createElement('span');
      item.className = 'activity-heatmap__weekday';
      item.textContent = i === 1 || i === 3 || i === 5 ? formatWeekday(addDays(sunday, i)) : '';
      container.appendChild(item);
    }
  }

  function renderEmpty(root, on) {
    var empty = root.querySelector('[data-activity-empty]');
    var viewport = root.querySelector('[data-activity-viewport]');
    var legend = root.querySelector('.activity-heatmap__legend');
    if (empty) empty.hidden = !on;
    if (viewport) viewport.hidden = on;
    if (legend) legend.hidden = on;
  }

  function clearDetail(root) {
    var detail = root.querySelector('[data-activity-detail]');
    root.querySelectorAll('.activity-heatmap__day[data-active="true"]').forEach(function (el) {
      el.removeAttribute('data-active');
    });
    if (detail) detail.hidden = true;
  }

  function renderDetail(root, posts, selectedDateKey) {
    var detailWindow = parseInt(root.getAttribute('data-detail-window-days'), 10);
    if (!Number.isFinite(detailWindow) || detailWindow < 0) detailWindow = 14;

    var selectedDate = dateFromKey(selectedDateKey);
    var detail = root.querySelector('[data-activity-detail]');
    var title = root.querySelector('[data-activity-detail-title]');
    var summary = root.querySelector('[data-activity-detail-summary]');
    var list = root.querySelector('[data-activity-detail-list]');
    if (!selectedDate || !detail || !title || !summary || !list) return;

    var byDate = groupByDate(posts);

    var windowPosts = [];
    for (var offset = -detailWindow; offset <= detailWindow; offset++) {
      var date = addDays(selectedDate, offset);
      var key = keyFromDate(date);
      var dayPosts = byDate.get(key) || [];
      dayPosts.forEach(function (post) {
        windowPosts.push({ date: key, post: post });
      });
    }

    var startKey = keyFromDate(addDays(selectedDate, -detailWindow));
    var endKey = keyFromDate(addDays(selectedDate, detailWindow));
    title.textContent = formatReadableDate(startKey) + ' – ' + formatReadableDate(endKey);
    summary.textContent = countLabel(root, windowPosts.length);
    list.innerHTML = '';

    if (!windowPosts.length) {
      var empty = document.createElement('div');
      empty.className = 'activity-heatmap__detail-empty';
      empty.textContent = getLabel(root, 'no-posts', 'No posts');
      list.appendChild(empty);
    } else {
      var probe = document.createElement('div');
      probe.className = 'timeline-item';
      probe.style.visibility = 'hidden';
      probe.style.position = 'absolute';
      probe.style.pointerEvents = 'none';
      document.body.appendChild(probe);
      var ITEM_WIDTH = probe.getBoundingClientRect().width;
      document.body.removeChild(probe);

      var MIN_GAP = 32;
      var PIXELS_PER_DAY = ITEM_WIDTH + MIN_GAP + 14;

      var firstPostTime = dateFromKey(windowPosts[0].date).getTime();
      var lastPostTime = dateFromKey(windowPosts[windowPosts.length - 1].date).getTime();
      var dataDuration = lastPostTime - firstPostTime;
      var distinctDays = (dataDuration / DAY_MS) + 1;
      var scrollWidth = distinctDays * PIXELS_PER_DAY;
      var windowDuration = dataDuration > 0 ? dataDuration : DAY_MS;
      var windowStartTime = firstPostTime;

      list.style.setProperty('--timeline-scroll-width', scrollWidth + 'px');

      var usableWidth = scrollWidth - ITEM_WIDTH;
      var STEP = (ITEM_WIDTH + MIN_GAP) / 2;
      var prevDate = null;
      var prevCenterX = -ITEM_WIDTH;
      var prevSide = 'below';

      windowPosts.forEach(function (item) {
        var postTime = dateFromKey(item.date).getTime();
        var ratio = (postTime - windowStartTime) / windowDuration;
        var idealX = (ITEM_WIDTH / 2) + ratio * usableWidth;
        var side;
        var centerX;

        if (item.date !== prevDate) {
          side = 'below';
          var prevRightEdge = prevCenterX + ITEM_WIDTH / 2;
          if (idealX - ITEM_WIDTH / 2 < prevRightEdge + MIN_GAP) {
            centerX = prevRightEdge + MIN_GAP + ITEM_WIDTH / 2;
          } else {
            centerX = idealX;
          }
        } else {
          side = prevSide === 'below' ? 'above' : 'below';
          centerX = prevCenterX + STEP;
        }

        prevDate = item.date;
        prevCenterX = centerX;
        prevSide = side;

        var timelineItem = document.createElement('a');
        timelineItem.className = 'timeline-item' + (side === 'above' ? ' timeline-item--above' : '');
        timelineItem.href = item.post.url || '#';
        timelineItem.style.left = centerX + 'px';

        var content = document.createElement('div');
        content.className = 'timeline-content';

        var dateSpan = document.createElement('span');
        dateSpan.className = 'timeline-date';
        dateSpan.textContent = formatReadableDate(item.date);
        content.appendChild(dateSpan);

        var title = document.createElement('span');
        title.className = 'timeline-title';
        title.textContent = item.post.title || '(untitled)';
        content.appendChild(title);

        var marker = document.createElement('div');
        marker.className = 'timeline-marker';

        if (side === 'above') {
          timelineItem.appendChild(content);
          timelineItem.appendChild(marker);
        } else {
          timelineItem.appendChild(marker);
          timelineItem.appendChild(content);
        }
        list.appendChild(timelineItem);
      });

      var finalScrollWidth = Math.max(scrollWidth, (prevCenterX + ITEM_WIDTH / 2) - MIN_GAP);
      list.style.setProperty('--timeline-scroll-width', finalScrollWidth + 'px');
    }

    detail.hidden = false;

    var listWrapper = list.parentNode.querySelector('.activity-heatmap__detail-list-wrap');
    if (!listWrapper) {
      listWrapper = document.createElement('div');
      listWrapper.className = 'activity-heatmap__detail-list-wrap';
      list.parentNode.insertBefore(listWrapper, list);
      listWrapper.appendChild(list);
    }

    var leftHint = listWrapper.querySelector('.activity-heatmap__detail-scroll-hint--left');
    var rightHint = listWrapper.querySelector('.activity-heatmap__detail-scroll-hint--right');
    if (!leftHint) {
      leftHint = document.createElement('div');
      leftHint.className = 'activity-heatmap__detail-scroll-hint activity-heatmap__detail-scroll-hint--left';
      var leftArrow = document.createElement('span');
      leftArrow.className = 'scroll-arrow';
      leftArrow.textContent = '‹';
      leftHint.appendChild(leftArrow);
      leftHint.addEventListener('click', function () {
        list.scrollBy({ left: -200, behavior: 'smooth' });
      });
      listWrapper.appendChild(leftHint);
    }
    if (!rightHint) {
      rightHint = document.createElement('div');
      rightHint.className = 'activity-heatmap__detail-scroll-hint activity-heatmap__detail-scroll-hint--right';
      var rightArrow = document.createElement('span');
      rightArrow.className = 'scroll-arrow';
      rightArrow.textContent = '›';
      rightHint.appendChild(rightArrow);
      rightHint.addEventListener('click', function () {
        list.scrollBy({ left: 200, behavior: 'smooth' });
      });
      listWrapper.appendChild(rightHint);
    }

    function updateScrollHints() {
      var maxScroll = list.scrollWidth - list.clientWidth;
      var canScroll = maxScroll > 1;
      leftHint.classList.toggle('activity-heatmap__detail-scroll-hint--visible', canScroll && list.scrollLeft > 8);
      rightHint.classList.toggle('activity-heatmap__detail-scroll-hint--visible', canScroll && list.scrollLeft < maxScroll - 8);
    }

    if (list._scrollHintHandler) {
      list.removeEventListener('scroll', list._scrollHintHandler);
    }
    list._scrollHintHandler = updateScrollHints;
    list.addEventListener('scroll', updateScrollHints, { passive: true });
    updateScrollHints();
  }

  function renderCalendar(root, state) {
    var cells = root.querySelector('[data-activity-cells]');
    var months = root.querySelector('[data-activity-months]');
    var years = root.querySelector('[data-activity-years]');
    var summary = root.querySelector('[data-activity-summary]');
    if (!cells || !months || !summary) return;

    var posts = state.posts;
    if (!posts.length) {
      summary.textContent = getLabel(root, 'no-posts', 'No posts');
      cells.innerHTML = '';
      months.innerHTML = '';
      if (years) years.innerHTML = '';
      return;
    }

    var byDate = groupByDate(posts);
    var maxCount = 0;
    byDate.forEach(function (items) {
      if (items.length > maxCount) maxCount = items.length;
    });

    var dates = Array.from(byDate.keys()).map(function (k) { return dateFromKey(k); }).filter(Boolean);
    var postMinDate = new Date(Math.min.apply(null, dates.map(function (d) { return d.getTime(); })));
    var maxDate = new Date();

    // Ensure at least one full past year is visible
    var oneYearAgo = new Date();
    oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
    var minDate = postMinDate < oneYearAgo ? postMinDate : oneYearAgo;

    var start = startOfWeek(minDate);
    var end = endOfWeek(maxDate);
    var dayCount = Math.round((end.getTime() - start.getTime()) / DAY_MS) + 1;
    var yearBoundaries = maxDate.getFullYear() - minDate.getFullYear();
    var totalMonths = (maxDate.getFullYear() - minDate.getFullYear()) * 12 + maxDate.getMonth() - minDate.getMonth() + 1;
    var monthBoundaries = Math.max(0, totalMonths - 1 - yearBoundaries);
    var weekCount = Math.ceil(dayCount / 7) + yearBoundaries + monthBoundaries;
    root.style.setProperty('--activity-week-count', String(weekCount));

    var minYear = minDate.getFullYear();
    var maxYear = maxDate.getFullYear();
    var yearRange = minYear === maxYear ? String(minYear) : minYear + '–' + maxYear;
    summary.textContent = countLabel(root, posts.length) + ' · ' + yearRange;

    cells.innerHTML = '';
    months.innerHTML = '';
    if (years) years.innerHTML = '';

    var seenMonths = new Set();
    var seenYears = new Set();
    var prevYear = null;
    for (var d = new Date(start.getTime()), index = 0; d <= end; d = addDays(d, 1), index++) {
      // Insert spacer column between years
      if (prevYear !== null && d.getFullYear() !== prevYear) {
        for (var s = 0; s < 7; s++) {
          var spacer = document.createElement('div');
          spacer.className = 'activity-heatmap__spacer';
          spacer.setAttribute('aria-hidden', 'true');
          cells.appendChild(spacer);
        }
        index += 7;
      }
      prevYear = d.getFullYear();

      if (d.getDate() === 1) {
        var monthKey = d.getFullYear() + '-' + d.getMonth();

        // Insert spacer column between months (skip first rendered month and January, which has year spacer)
        if (seenMonths.size > 0 && d.getMonth() !== 0) {
          for (var ms = 0; ms < 7; ms++) {
            var mSpacer = document.createElement('div');
            mSpacer.className = 'activity-heatmap__spacer';
            mSpacer.setAttribute('aria-hidden', 'true');
            cells.appendChild(mSpacer);
          }
          index += 7;
        }

        // Year label at first month of each year
        if (years && !seenYears.has(d.getFullYear())) {
          seenYears.add(d.getFullYear());
          var yearLabel = document.createElement('span');
          yearLabel.className = 'activity-heatmap__year';
          yearLabel.style.gridColumn = String(Math.floor(index / 7) + 1);
          yearLabel.textContent = String(d.getFullYear());
          years.appendChild(yearLabel);
        }

        if (!seenMonths.has(monthKey)) {
          seenMonths.add(monthKey);
          var month = document.createElement('span');
          month.className = 'activity-heatmap__month';
          month.style.gridColumn = String(Math.floor(index / 7) + 1);
          month.textContent = formatMonth(d);
          months.appendChild(month);
        }
      }

      var key = keyFromDate(d);
      var dayPosts = byDate.get(key) || [];
      var count = dayPosts.length;
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'activity-heatmap__day';
      button.setAttribute('data-date', key);
      button.setAttribute('data-count', String(count));
      button.setAttribute('data-level', String(levelFor(count, maxCount)));

      var label = formatReadableDate(key) + ': ' + countLabel(root, count);
      button.setAttribute('aria-label', label);
      button.setAttribute('title', label);
      if (state.selectedDate === key) button.setAttribute('data-active', 'true');
      button.addEventListener('click', function (event) {
        var date = event.currentTarget.getAttribute('data-date');
        state.selectedDate = date;
        root.querySelectorAll('.activity-heatmap__day[data-active="true"]').forEach(function (el) {
          el.removeAttribute('data-active');
        });
        event.currentTarget.setAttribute('data-active', 'true');
        renderDetail(root, state.posts, date);
      });

      cells.appendChild(button);
    }

    if (state.selectedDate) {
      var selected = cells.querySelector('[data-date="' + state.selectedDate + '"]');
      if (selected) {
        selected.setAttribute('data-active', 'true');
      } else {
        clearDetail(root);
        state.selectedDate = '';
      }
    }

    var viewport = root.querySelector('[data-activity-viewport]');
    if (viewport) {
      viewport.scrollLeft = viewport.scrollWidth;
    }
  }

  function initHeatmap(root) {
    var dataEl = root.querySelector('[data-activity-data]');
    if (!dataEl) return;

    var payload;
    try {
      payload = JSON.parse(dataEl.textContent || '{}');
    } catch (e) {
      payload = {};
    }

    var posts = Array.isArray(payload.posts) ? payload.posts.filter(function (post) {
      return post && /^\d{4}-\d{2}-\d{2}$/.test(post.date || '');
    }) : [];

    if (!posts.length) {
      renderEmpty(root, true);
      var summary = root.querySelector('[data-activity-summary]');
      if (summary) summary.textContent = getLabel(root, 'no-posts', 'No posts');
      return;
    }

    renderEmpty(root, false);
    initWeekdays(root);

    // Set up viewport wrapper before rendering so DOM move doesn't reset scroll
    var viewport = root.querySelector('[data-activity-viewport]');
    if (viewport) {
      var vpWrapper = document.createElement('div');
      vpWrapper.className = 'activity-heatmap__viewport-wrap';
      viewport.parentNode.insertBefore(vpWrapper, viewport);
      vpWrapper.appendChild(viewport);

      var vpLeft = document.createElement('div');
      vpLeft.className = 'activity-heatmap__viewport-scroll-hint activity-heatmap__viewport-scroll-hint--left';
      var vpLa = document.createElement('span');
      vpLa.className = 'scroll-arrow';
      vpLa.textContent = '‹';
      vpLeft.appendChild(vpLa);
      vpLeft.addEventListener('click', function () {
        viewport.scrollBy({ left: -200, behavior: 'smooth' });
      });
      vpWrapper.appendChild(vpLeft);

      var vpRight = document.createElement('div');
      vpRight.className = 'activity-heatmap__viewport-scroll-hint activity-heatmap__viewport-scroll-hint--right';
      var vpRa = document.createElement('span');
      vpRa.className = 'scroll-arrow';
      vpRa.textContent = '›';
      vpRight.appendChild(vpRa);
      vpRight.addEventListener('click', function () {
        viewport.scrollBy({ left: 200, behavior: 'smooth' });
      });
      vpWrapper.appendChild(vpRight);

      function updateVpScrollHints() {
        var maxScroll = viewport.scrollWidth - viewport.clientWidth;
        var canScroll = maxScroll > 1;
        vpLeft.classList.toggle('activity-heatmap__viewport-scroll-hint--visible', canScroll && viewport.scrollLeft > 8);
        vpRight.classList.toggle('activity-heatmap__viewport-scroll-hint--visible', canScroll && viewport.scrollLeft < maxScroll - 8);
      }

      viewport.addEventListener('scroll', updateVpScrollHints, { passive: true });
    }

    var state = {
      posts: posts,
      selectedDate: ''
    };

    var detailClose = root.querySelector('[data-activity-detail-close]');
    if (detailClose) {
      detailClose.addEventListener('click', function () {
        state.selectedDate = '';
        clearDetail(root);
      });
    }

    renderCalendar(root, state);

    // Auto-expand the most recent date
    var cells = root.querySelectorAll('.activity-heatmap__day');
    var latestKey = '';
    var latestCell = null;
    cells.forEach(function (cell) {
      var key = cell.getAttribute('data-date');
      if (key && parseInt(cell.getAttribute('data-count'), 10) > 0 && key > latestKey) {
        latestKey = key;
        latestCell = cell;
      }
    });
    if (latestCell) latestCell.click();

    // Ensure viewport stays at right after detail expansion may cause reflow
    if (viewport) {
      viewport.scrollLeft = viewport.scrollWidth;
      updateVpScrollHints();
    }
  }

  function initAll() {
    document.querySelectorAll('.activity-heatmap').forEach(initHeatmap);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }
})();
