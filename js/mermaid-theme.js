/**
 * Mermaid：随 MDUI 亮/暗与系统配色切换主题，暗色下提高对比度；支持重复 run。
 */
(function () {
  var HIGH_CONTRAST_DARK = {
    primaryTextColor: '#f2f2f8',
    secondaryTextColor: '#e0e0ec',
    tertiaryTextColor: '#c8c8d8',
    lineColor: '#c2c8da',
    textColor: '#f2f2f8',
    mainBkg: '#252530',
    nodeBorder: '#9aa6c4',
    clusterBkg: 'rgba(255,255,255,0.1)',
    clusterBorder: '#8a94b0',
    titleColor: '#fafafc',
    edgeLabelBackground: '#1a1a22',
    actorBorder: '#aab4d0',
    actorBkg: '#2c2c38',
    actorTextColor: '#f2f2f8',
    signalColor: '#dde0ec',
    signalTextColor: '#eceef6',
    labelBoxBkgColor: '#2c2c38',
    labelBoxBorderColor: '#9aa6c4',
    labelTextColor: '#f2f2f8',
    loopTextColor: '#f2f2f8',
    activationBorderColor: '#aab4d0',
    activationBkgColor: '#38384a',
    sequenceNumberColor: '#1a1a22',
    gridColor: 'rgba(200,205,230,0.22)',
    sectionBkgColor2: '#2e2e3a'
  };

  function isEffectivelyDark() {
    var el = document.documentElement;
    if (el.classList.contains('mdui-theme-dark')) return true;
    if (el.classList.contains('mdui-theme-light')) return false;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  function buildOpts() {
    var base = window.__HEXO_MERMAID__ || {};
    var o = Object.assign({}, base);
    o.startOnLoad = false;
    var dark = isEffectivelyDark();
    o.theme = dark ? 'dark' : (base.theme || 'default');
    if (dark) {
      o.themeVariables = Object.assign({}, base.themeVariables || {}, HIGH_CONTRAST_DARK);
    } else if (base.themeVariables && typeof base.themeVariables === 'object') {
      o.themeVariables = Object.assign({}, base.themeVariables);
    } else {
      delete o.themeVariables;
    }
    return o;
  }

  function ensureSources(nodes) {
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (!el.dataset.mermaidSrc) el.dataset.mermaidSrc = el.textContent;
    }
  }

  function resetNodes(nodes) {
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      var src = el.dataset.mermaidSrc;
      if (src == null) continue;
      el.innerHTML = '';
      el.textContent = src;
      el.removeAttribute('data-processed');
      el.removeAttribute('data-mermaid-processed');
    }
  }

  async function refreshMermaid() {
    if (!window.mermaid || !window.__HEXO_MERMAID__) return;
    var nodes = document.querySelectorAll('pre.mermaid');
    if (!nodes.length) return;
    ensureSources(nodes);
    resetNodes(nodes);
    mermaid.initialize(buildOpts());
    await mermaid.run({ querySelector: 'pre.mermaid' });
  }

  window.__mermaidRerender = refreshMermaid;

  function boot() {
    void refreshMermaid();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
