(function () {
  'use strict';

  function initPostTocScrollSpy() {
    var allLinks = Array.prototype.slice.call(
      document.querySelectorAll('.post-toc-scroll .post-toc-link[href^="#"]')
    );

    if (!allLinks.length) return;

    var recordsByHeading = new Map();

    allLinks.forEach(function (link) {
      var rawId = link.getAttribute('href').slice(1);
      var id;

      try {
        id = decodeURIComponent(rawId);
      } catch (error) {
        id = rawId;
      }

      if (!id) return;

      var heading = document.getElementById(id);
      if (!heading || !/^H[23]$/.test(heading.tagName)) return;

      if (!recordsByHeading.has(heading)) {
        recordsByHeading.set(heading, { heading: heading, links: [] });
      }

      recordsByHeading.get(heading).links.push(link);
    });

    var records = Array.from(recordsByHeading.values());
    if (!records.length) return;

    var positions = [];
    var activeRecord = null;
    var frameId = 0;
    var resizeObserver = null;

    function activationOffset() {
      var stickySidebar = document.querySelector('.sticky-sidebar');
      var stickyTop = stickySidebar ? parseFloat(getComputedStyle(stickySidebar).top) : NaN;

      if (Number.isFinite(stickyTop)) return stickyTop + 8;

      var topBar = document.querySelector('.app-top-bar');
      return (topBar ? topBar.getBoundingClientRect().bottom : 64) + 8;
    }

    function directItemLink(item) {
      if (!item) return null;

      for (var i = 0; i < item.children.length; i += 1) {
        if (item.children[i].classList.contains('post-toc-link')) return item.children[i];
      }

      return null;
    }

    function ancestorLink(link) {
      var item = link.closest('.post-toc-item');
      if (!item || !item.parentElement) return null;

      return directItemLink(item.parentElement.closest('.post-toc-item'));
    }

    function keepLinkVisible(link) {
      var scroller = link.closest('.post-toc-scroll');
      if (!scroller || scroller.clientHeight < 1) return;

      var scrollerRect = scroller.getBoundingClientRect();
      var linkRect = link.getBoundingClientRect();
      var breathingRoom = 16;

      if (linkRect.top < scrollerRect.top + breathingRoom) {
        scroller.scrollTop -= scrollerRect.top + breathingRoom - linkRect.top;
      } else if (linkRect.bottom > scrollerRect.bottom - breathingRoom) {
        scroller.scrollTop += linkRect.bottom - (scrollerRect.bottom - breathingRoom);
      }
    }

    function keepActiveLinksVisible() {
      if (!activeRecord) return;
      activeRecord.links.forEach(keepLinkVisible);
    }

    function applyActiveState(nextRecord) {
      if (!nextRecord) return;

      if (nextRecord === activeRecord) {
        requestAnimationFrame(keepActiveLinksVisible);
        return;
      }

      allLinks.forEach(function (link) {
        link.classList.remove('is-active', 'is-active-ancestor');
        link.removeAttribute('aria-current');
      });

      nextRecord.links.forEach(function (link) {
        link.classList.add('is-active');
        link.setAttribute('aria-current', 'location');

        var parentLink = ancestorLink(link);
        if (parentLink) parentLink.classList.add('is-active-ancestor');
      });

      activeRecord = nextRecord;
      requestAnimationFrame(function () {
        keepActiveLinksVisible();
        requestAnimationFrame(keepActiveLinksVisible);
      });
    }

    function updateActiveHeading() {
      frameId = 0;
      if (!positions.length) return;

      var threshold = window.scrollY + activationOffset();
      var index = 0;

      for (var i = 0; i < positions.length; i += 1) {
        if (positions[i].top <= threshold) index = i;
        else break;
      }

      var documentHeight = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
      if (window.scrollY + window.innerHeight >= documentHeight - 2) index = positions.length - 1;

      applyActiveState(positions[index].record);
    }

    function requestUpdate() {
      if (!frameId) frameId = requestAnimationFrame(updateActiveHeading);
    }

    function recalculate() {
      positions = records
        .map(function (record) {
          return {
            record: record,
            top: record.heading.getBoundingClientRect().top + window.scrollY
          };
        })
        .sort(function (a, b) {
          return a.top - b.top;
        });

      requestUpdate();
    }

    window.addEventListener('scroll', requestUpdate, { passive: true });
    window.addEventListener('resize', recalculate, { passive: true });
    window.addEventListener('load', recalculate, { once: true });
    window.addEventListener('hashchange', requestUpdate);

    document.addEventListener('click', function (event) {
      if (event.target.closest('.post-toc-scroll .post-toc-link[href^="#"]')) {
        requestAnimationFrame(requestUpdate);
      }
    });

    var drawer = document.getElementById('site-category-drawer');
    if (drawer) {
      drawer.addEventListener('open', function () {
        requestAnimationFrame(keepActiveLinksVisible);
      });
      drawer.addEventListener('opened', function () {
        requestAnimationFrame(keepActiveLinksVisible);
      });

      new MutationObserver(function () {
        requestAnimationFrame(keepActiveLinksVisible);
      }).observe(drawer, { attributes: true, attributeFilter: ['open'] });
    }

    if ('ResizeObserver' in window) {
      var articleBodies = document.querySelectorAll('.post-page-card__body, .page-page__body');
      if (articleBodies.length) {
        resizeObserver = new ResizeObserver(recalculate);
        articleBodies.forEach(function (el) {
          resizeObserver.observe(el);
        });
      }
    }

    recalculate();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPostTocScrollSpy, { once: true });
  } else {
    initPostTocScrollSpy();
  }
})();
