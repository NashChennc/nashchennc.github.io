/**
 * mdui-card（带 href）在 Shadow 内用 <a class="link"> 包 slot，且为 display:inline-block，
 * 外部 flex 无法作用在正文/meta 上。将内部链接改为纵向 flex，使 .note-card-body + margin-top:auto 生效。
 */
(function () {
  function patchCard(card) {
    if (!card.classList.contains('note-card')) return;
    var root = card.shadowRoot;
    if (!root) return;
    var link = root.querySelector('a._a.link');
    if (!link || link.dataset.mdNoteCardLayout === '1') return;
    link.style.display = 'flex';
    link.style.flexDirection = 'column';
    link.style.alignItems = 'stretch';
    link.style.justifyContent = 'flex-start';
    link.style.flex = '1 1 auto';
    link.style.minHeight = '0';
    link.style.width = '100%';
    link.style.boxSizing = 'border-box';
    link.dataset.mdNoteCardLayout = '1';
  }

  function run() {
    document.querySelectorAll('mdui-card.note-card').forEach(patchCard);
  }

  function schedule() {
    requestAnimationFrame(run);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', schedule);
  } else {
    schedule();
  }

  if (typeof customElements !== 'undefined' && customElements.whenDefined) {
    customElements.whenDefined('mdui-card').then(schedule).catch(function () {});
  }

  setTimeout(schedule, 0);
})();
