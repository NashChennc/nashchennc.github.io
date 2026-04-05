/**
 * 顶栏亮/暗主题切换：localStorage key mdblog-theme；Shift+点击恢复跟随系统。
 * 主题 DOM 由 mdui.setTheme 管理；无 mdui 时回退为手动 class。
 */
(function () {
  var STORAGE_KEY = 'mdblog-theme';

  function getStored() {
    try {
      var t = localStorage.getItem(STORAGE_KEY);
      if (t === 'light' || t === 'dark') return t;
    } catch (e) {}
    return null;
  }

  function setStored(theme) {
    try {
      if (theme === 'light' || theme === 'dark') localStorage.setItem(STORAGE_KEY, theme);
      else localStorage.removeItem(STORAGE_KEY);
    } catch (e) {}
  }

  function themeModeFromMdui() {
    if (typeof mdui !== 'undefined' && typeof mdui.getTheme === 'function') {
      return mdui.getTheme();
    }
    var el = document.documentElement;
    if (el.classList.contains('mdui-theme-dark')) return 'dark';
    if (el.classList.contains('mdui-theme-light')) return 'light';
    if (el.classList.contains('mdui-theme-auto')) return 'auto';
    return 'light';
  }

  function isEffectivelyDark() {
    var mode = themeModeFromMdui();
    if (mode === 'dark') return true;
    if (mode === 'light') return false;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  function applyHtmlClass(theme) {
    var el = document.documentElement;
    el.classList.remove('mdui-theme-auto', 'mdui-theme-light', 'mdui-theme-dark');
    if (theme === 'light' || theme === 'dark') el.classList.add('mdui-theme-' + theme);
    else el.classList.add('mdui-theme-auto');
  }

  function setMduiTheme(theme) {
    if (typeof mdui !== 'undefined' && typeof mdui.setTheme === 'function') {
      mdui.setTheme(theme);
    } else {
      applyHtmlClass(theme);
    }
  }

  /** 与 <head> 防闪烁脚本一致：用 localStorage 同步 MDUI 状态。 */
  function syncThemeFromStorage() {
    var t = getStored();
    var next = t === 'light' || t === 'dark' ? t : 'auto';
    setMduiTheme(next);
  }

  function notifyMermaidRerender() {
    if (typeof window.__mermaidRerender === 'function') {
      void window.__mermaidRerender();
    }
  }

  function updateToggleButton(btn) {
    if (!btn) return;
    var dark = isEffectivelyDark();
    var labelDark = btn.getAttribute('data-label-dark') || 'Dark mode';
    var labelLight = btn.getAttribute('data-label-light') || 'Light mode';
    if (dark) {
      btn.setAttribute('aria-label', labelLight);
      btn.setAttribute('icon', 'light_mode');
    } else {
      btn.setAttribute('aria-label', labelDark);
      btn.setAttribute('icon', 'dark_mode');
    }
  }

  function init() {
    syncThemeFromStorage();

    var btn = document.getElementById('theme-toggle');
    if (!btn) return;

    updateToggleButton(btn);

    var mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
    function onSchemeChange() {
      if (themeModeFromMdui() === 'auto') {
        updateToggleButton(btn);
        notifyMermaidRerender();
      }
    }
    if (mq && mq.addEventListener) {
      mq.addEventListener('change', onSchemeChange);
    } else if (mq && mq.addListener) {
      mq.addListener(onSchemeChange);
    }

    btn.addEventListener('click', function (ev) {
      if (ev.shiftKey) {
        setStored(null);
        setMduiTheme('auto');
        updateToggleButton(btn);
        notifyMermaidRerender();
        return;
      }

      var dark = isEffectivelyDark();
      var next = dark ? 'light' : 'dark';
      setStored(next);
      setMduiTheme(next);
      updateToggleButton(btn);
      notifyMermaidRerender();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
