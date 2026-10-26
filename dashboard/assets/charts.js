// Lightweight d3 charts for the live dashboard.
//
// Lines are drawn on a <canvas> (fast for thousands of points times dozens of
// series); axes, grid, reference lines, the zoom brush and the hover marker are
// SVG on top. Drag a box to zoom (horizontal / vertical drags zoom one axis),
// pan with the pan button or shift-drag, Ctrl/⌘ + wheel to zoom at the cursor, double-click to reset.
(function () {
  'use strict';

  const DASH = {
    solid: [], dot: [1.5, 3], dash: [6, 4], longdash: [11, 5],
    dashdot: [6, 3, 1.5, 3], longdashdot: [11, 4, 1.5, 4],
  };
  const SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  const sup = (n) => String(n).split('').map((c) => SUP[c] || c).join('');

  // Text with <sub>…</sub> / <sup>…</sup> markup, as HTML (tooltips, legend)
  // or as SVG tspans (axes). Unicode superscript digits render unevenly in
  // most fonts, so exponents use real raised text instead.
  const tagRe = /<(sub|sup)>(.*?)<\/\1>/g;
  function rich(sel, str) {
    sel.text(null);
    str = str == null ? '' : String(str);
    let last = 0, m, cur = 0;                 // cur: baseline offset in em
    const put = (t, target, small) => {
      if (!t) return;
      const ts = sel.append('tspan').text(t);
      if (target !== cur) ts.attr('dy', (target - cur) + 'em');
      if (small) ts.attr('font-size', '0.75em');
      cur = target;
    };
    tagRe.lastIndex = 0;
    while ((m = tagRe.exec(str))) {
      put(str.slice(last, m.index), 0, false);
      put(m[2], m[1] === 'sup' ? -0.45 : 0.3, true);
      last = tagRe.lastIndex;
    }
    put(str.slice(last), 0, false);
  }
  const richAll = (sel) => sel.each(function () { rich(d3.select(this), this.__rich ?? this.textContent); });

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // a×10ⁿ for very small or large numbers, plain otherwise.
  function fmtNum(v, digits = 3) {
    if (v === null || v === undefined || !isFinite(v)) return '–';
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) {
      const e = Math.floor(Math.log10(a));
      const m = +(v / Math.pow(10, e)).toPrecision(digits);
      return (m === 1 ? '' : m === -1 ? '−' : String(m).replace('-', '−') + '×') + '10<sup>' + String(e).replace('-', '−') + '</sup>';
    }
    return String(+v.toPrecision(digits)).replace('-', '−');
  }
  function tickFormatter(scale, log) {
    if (log) {
      return (v) => {
        const e = Math.log10(v);
        return Math.abs(e - Math.round(e)) < 1e-9 ? '10<sup>' + String(Math.round(e)).replace('-', '−') + '</sup>' : fmtNum(v, 2);
      };
    }
    const ticks = scale.ticks(6);
    const span = Math.max(...ticks.map(Math.abs));
    // Each tick carries its own exponent (2×10⁴, 4×10⁴, …, 10⁵) rather than a shared one.
    if (span >= 1e5 || (span < 1e-3 && span > 0)) return (v) => fmtNum(v, 3);
    const f = d3.format('~g');
    return (v) => f(v).replace('-', '−');
  }

  // ---------------------------------------------------------------------------
  // Export: every chart can be saved as SVG or PDF, as it looks on screen
  // (theme colours, zoom, hidden series). Curves drawn on the canvas are
  // rebuilt as SVG paths; the heatmap is embedded as an image.
  // ---------------------------------------------------------------------------
  const NS = 'http://www.w3.org/2000/svg';
  const EXPORT_FONT = "Inter, 'DejaVu Sans', Helvetica, Arial, sans-serif";
  const r2 = (v) => Math.round(v * 100) / 100;
  const xmlEsc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const plain = (v) => String(v == null ? '' : v).replace(/<[^>]+>/g, '');
  let measureCtx = null;
  const textWidth = (str, size = 11, weight = 400) => {
    measureCtx = measureCtx || document.createElement('canvas').getContext('2d');
    measureCtx.font = `${weight} ${size}px ${EXPORT_FONT}`;
    return measureCtx.measureText(plain(str)).width;
  };
  // Clone a live SVG layer with the styles the stylesheet gives it written
  // inline (fonts, fills, strokes), em units resolved, interaction layers dropped.
  function cloneLayer(svg) {
    const g = document.createElementNS(NS, 'g');
    const src = [...svg.childNodes].filter((n) => n.nodeType === 1 && !/(^| )(overlay|brush|hover)( |$)/.test(n.getAttribute('class') || ''));
    for (const n of src) {
      const c = n.cloneNode(true);
      const a = [n, ...n.querySelectorAll('*')], b = [c, ...c.querySelectorAll('*')];
      a.forEach((el, i) => {
        const t = b[i], cs = getComputedStyle(el), tag = el.tagName.toLowerCase();
        if (tag === 'text') {
          t.setAttribute('font-size', cs.fontSize); t.setAttribute('font-weight', +cs.fontWeight >= 600 ? 'bold' : 'normal');
          if (!t.getAttribute('fill')) t.setAttribute('fill', cs.fill);
          const dy = el.getAttribute('dy');
          if (dy && dy.endsWith('em')) t.setAttribute('dy', r2(parseFloat(dy) * parseFloat(cs.fontSize)) + '');
        } else if (tag === 'tspan') {
          const px = parseFloat(cs.fontSize);
          if (el.getAttribute('font-size')) t.setAttribute('font-size', px + 'px');
          const dy = el.getAttribute('dy');
          if (dy && dy.endsWith('em')) t.setAttribute('dy', r2(parseFloat(dy) * parseFloat(getComputedStyle(el.parentNode).fontSize)) + '');
        } else if (tag === 'line' || tag === 'path') {
          if (!t.getAttribute('stroke') && cs.stroke !== 'none') t.setAttribute('stroke', cs.stroke);
          if (!t.getAttribute('fill')) t.setAttribute('fill', cs.fill === 'rgb(0, 0, 0)' && tag === 'path' ? 'none' : cs.fill);
        }
        t.removeAttribute('class');
        if (t.getAttribute('font-family') === 'sans-serif') t.removeAttribute('font-family');   // d3's axis default
      });
      g.appendChild(c);
    }
    return new XMLSerializer().serializeToString(g).replace(/ xmlns="[^"]*"/g, '');
  }
  function svgDoc(w, h, body) {
    return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="${NS}" xmlns:xlink="http://www.w3.org/1999/xlink" width="${r2(w)}" height="${r2(h)}" viewBox="0 0 ${r2(w)} ${r2(h)}" font-family="${xmlEsc(EXPORT_FONT)}">` +
      `<rect width="${r2(w)}" height="${r2(h)}" fill="${cssVar('--surface-1') || '#fff'}"/>${body}</svg>`;
  }
  function save(blob, name) {
    const href = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), { href, download: name });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(href), 10000);
  }
  // jsPDF, svg2pdf.js and the DejaVu Sans subsets (which carry ψ, μ, ², ₁₀, …)
  // load the first time a PDF is asked for.
  let pdfReady = null;
  function loadPdf() {
    if (pdfReady) return pdfReady;
    const script = (src) => new Promise((ok, ko) => {
      const el = document.createElement('script'); el.src = src; el.onload = ok;
      el.onerror = () => ko(new Error(src + ' could not be loaded'));
      document.head.appendChild(el);
    });
    const font = (f) => fetch('/assets/' + f).then((r) => { if (!r.ok) throw new Error(f + ' not found'); return r.arrayBuffer(); })
      .then((buf) => { let s = ''; const u = new Uint8Array(buf); for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); });
    // svg2pdf measures text in the browser: give it the same font the PDF embeds.
    const face = document.createElement('style');
    face.textContent = [['', 'normal', 'normal'], ['-bold', 'bold', 'normal'], ['-oblique', 'normal', 'italic']]
      .map(([f, wt, st]) => `@font-face { font-family: 'DejaVuSans'; font-weight: ${wt}; font-style: ${st}; src: url('/assets/dejavu-sans${f}-subset.ttf') format('truetype'); }`).join('\n');
    document.head.appendChild(face);
    const faces = () => Promise.all(['normal 11px', 'bold 11px', 'italic 11px'].map((f) => document.fonts.load(f + ' DejaVuSans')));
    pdfReady = script('/assets/jspdf.umd.min.js').then(() => script('/assets/svg2pdf.umd.min.js'))
      .then(() => Promise.all(['dejavu-sans-subset.ttf', 'dejavu-sans-bold-subset.ttf', 'dejavu-sans-oblique-subset.ttf'].map(font)))
      .then(async (f) => { await faces(); return f; })
      .catch((e) => { pdfReady = null; throw e; });
    return pdfReady;
  }
  async function savePdf(svgText, w, h, name) {
    const fonts = await loadPdf();
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ orientation: w > h ? 'landscape' : 'portrait', unit: 'pt', format: [w, h], compress: true });
    [['normal', 0], ['bold', 1], ['italic', 2], ['bolditalic', 2]].forEach(([style, i]) => {
      const f = `DejaVuSans-${style}.ttf`;
      doc.addFileToVFS(f, fonts[i]); doc.addFont(f, 'DejaVuSans', style);
    });
    doc.setFont('DejaVuSans', 'normal');
    const el = new DOMParser().parseFromString(svgText.replace(/font-family="[^"]*"/g, 'font-family="DejaVuSans"'), 'image/svg+xml').documentElement;
    const host = document.createElement('div');
    host.style.cssText = 'position:fixed;left:-10000px;top:0;visibility:hidden';
    host.appendChild(el); document.body.appendChild(host);
    try { await doc.svg(el, { x: 0, y: 0, width: w, height: h }); } finally { host.remove(); }
    save(doc.output('blob'), name + '.pdf');
  }
  async function exportChart(chart, fmt) {
    const { svg, w, h } = chart.toSVG();
    const base = (D3Charts.exportName && D3Charts.exportName(chart.el)) || 'plot';
    if (fmt === 'svg') save(new Blob([svg], { type: 'image/svg+xml' }), base + '.svg');
    else await savePdf(svg, w, h, base);
  }
  // A download button in the chart's tool strip, with an SVG / PDF menu.
  function addExport(tools, chart) {
    const wrap = tools.append('div').attr('class', 'd3export');
    const btn = wrap.append('button').attr('class', 'mode').attr('type', 'button').attr('title', 'Download as SVG or PDF').attr('aria-label', 'Download plot')
      .attr('aria-haspopup', 'menu').attr('aria-expanded', 'false')
      .html('<svg viewBox="0 0 24 24"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>');
    const menu = wrap.append('div').attr('class', 'd3menu').attr('role', 'menu').attr('hidden', '');
    const close = () => { menu.attr('hidden', ''); btn.attr('aria-expanded', 'false'); document.removeEventListener('pointerdown', outside, true); };
    const outside = (ev) => { if (!wrap.node().contains(ev.target)) close(); };
    btn.on('click', () => {
      if (menu.attr('hidden') === null) { close(); return; }
      menu.attr('hidden', null); btn.attr('aria-expanded', 'true');
      document.addEventListener('pointerdown', outside, true);
    });
    menu.selectAll('button').data([['svg', 'SVG', 'vector, editable'], ['pdf', 'PDF', 'vector, for papers']]).join('button')
      .attr('type', 'button').attr('role', 'menuitem')
      .html((d) => `<b>${d[1]}</b><span>${d[2]}</span>`)
      .on('click', async (ev, d) => {
        close(); wrap.classed('busy', true);
        try { await exportChart(chart, d[0]); }
        catch (e) { (D3Charts.onError || console.error)(e); }
        wrap.classed('busy', false);
      });
    wrap.on('keydown', (ev) => { if (ev.key === 'Escape') { close(); btn.node().focus(); } });
  }

  // ---------------------------------------------------------------------------
  // Line chart
  // ---------------------------------------------------------------------------
  class LineChart {
    constructor(el, opts = {}) {
      this.el = el;
      this.opts = Object.assign({ hover: 'x', legend: true }, opts);
      this.hidden = new Set();
      this.view = null;          // zoomed x range in data units, or null
      this.yView = null;         // zoomed y range in data units, or null
      this.uirev = null;
      this.hovered = null;
      this.margin = { top: 10, right: 14, bottom: 40, left: 64 };

      el.classList.add('d3chart');
      el.innerHTML = '';
      this.area = d3.select(el).append('div').attr('class', 'd3area');
      // Grid on its own SVG under the canvas so the curves paint over it.
      this.bg = this.area.append('svg').attr('class', 'bg');
      this.gGrid = this.bg.append('g').attr('class', 'grid');
      this.canvas = this.area.append('canvas').node();
      this.svg = this.area.append('svg');
      this.gRefs = this.svg.append('g').attr('class', 'refs');
      this.gX = this.svg.append('g').attr('class', 'axis x');
      this.gY = this.svg.append('g').attr('class', 'axis y');
      this.xTitle = this.svg.append('text').attr('class', 'axis-title').attr('text-anchor', 'middle');
      this.yTitle = this.svg.append('text').attr('class', 'axis-title').attr('text-anchor', 'middle');
      this.gHover = this.svg.append('g').attr('class', 'hover');
      this.brush = this.svg.append('rect').attr('class', 'brush').attr('visibility', 'hidden');
      this.overlay = this.svg.append('rect').attr('class', 'overlay');
      this.tip = this.area.append('div').attr('class', 'd3tip');
      // Drag mode: box zoom or pan (holding Shift switches to the other one).
      this.dragMode = 'zoom';
      const tools = this.area.append('div').attr('class', 'd3tools');
      this.resetBtn = tools.append('button').attr('class', 'd3reset').text('Reset zoom')
        .on('click', () => this.resetZoom());
      const ICON = {
        zoom: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4.2-4.2M11 8v6M8 11h6"/>',
        pan: '<path d="M12 2v20M2 12h20M12 2l-3 3M12 2l3 3M12 22l-3-3M12 22l3-3M2 12l3-3M2 12l3 3M22 12l-3-3M22 12l-3 3"/>',
      };
      const TITLE = { zoom: 'Zoom: drag a box (Shift-drag pans)', pan: 'Pan: drag to move (Shift-drag zooms)' };
      this.modeBtns = tools.selectAll('button.mode').data(['zoom', 'pan']).join('button')
        .attr('class', 'mode').attr('title', (d) => TITLE[d]).attr('aria-label', (d) => TITLE[d])
        .html((d) => `<svg viewBox="0 0 24 24">${ICON[d]}</svg>`)
        .classed('on', (d) => d === this.dragMode)
        .on('click', (ev, d) => {
          this.dragMode = d;
          this.modeBtns.classed('on', (k) => k === d);
          this.overlay.classed('panmode', d === 'pan');
        });
      addExport(tools, this);
      this.legend = this.opts.legend ? d3.select(el).append('div').attr('class', 'd3legend') : null;

      this._bindPointer();
      this._ro = new ResizeObserver(() => this.draw());
      this._ro.observe(el);
    }

    update(spec) {
      if (spec.uirev !== undefined && spec.uirev !== this.uirev) {
        this.view = null; this.yView = null;
        this.uirev = spec.uirev;
      }
      this.spec = spec;
      this._renderLegend();
      this.draw();
    }

    // -- layout and scales ------------------------------------------------------
    _size() {
      const r = this.area.node().getBoundingClientRect();
      return { w: Math.max(50, r.width), h: Math.max(50, r.height) };
    }

    _visibleSeries() {
      return (this.spec.series || []).filter((s) => !this.hidden.has(s.name) && s.y && s.y.length);
    }

    _scales(w, h) {
      const m = this.margin, sp = this.spec;
      const series = this._visibleSeries();
      let x0 = Infinity, x1 = -Infinity;
      for (const s of series) for (const v of s.x) if (v !== null && isFinite(v)) { if (v < x0) x0 = v; if (v > x1) x1 = v; }
      if (!isFinite(x0)) { x0 = 0; x1 = 1; }
      if (x0 === x1) { x0 -= 0.5; x1 += 0.5; }
      const xd = this.view || [x0, x1];
      const x = d3.scaleLinear().domain(xd).range(sp.xReversed ? [w - m.right, m.left] : [m.left, w - m.right]);

      let yd = this.yView || sp.yRange;
      if (!yd) {
        let lo = Infinity, hi = -Infinity;
        const [a, b] = [Math.min(...xd), Math.max(...xd)];
        for (const s of series) {
          const xs = s.x, ys = s.y;
          for (let i = 0; i < ys.length; i++) {
            const v = ys[i], xv = xs[i];
            if (v === null || !isFinite(v) || xv === null || xv < a || xv > b) continue;
            if (sp.yLog && v <= 0) continue;
            if (v < lo) lo = v; if (v > hi) hi = v;
          }
        }
        for (const r of sp.hlines || []) {
          if (r.inRange === false || !isFinite(r.y) || (sp.yLog && r.y <= 0)) continue;
          if (r.y < lo) lo = r.y; if (r.y > hi) hi = r.y;
        }
        if (!isFinite(lo)) { lo = sp.yLog ? 1e-10 : 0; hi = sp.yLog ? 1 : 1; }
        if (sp.yLog) {
          const l0 = Math.log10(lo), l1 = Math.log10(hi), pad = Math.max(0.05 * (l1 - l0), 0.1);
          yd = [Math.pow(10, l0 - pad), Math.pow(10, l1 + pad)];
        } else {
          const pad = (hi - lo) * 0.06 || Math.abs(hi) * 0.1 || 1;
          yd = [lo - pad, hi + pad];
        }
      }
      const y = (sp.yLog ? d3.scaleLog() : d3.scaleLinear()).domain(yd).range([h - m.bottom, m.top]);
      return { x, y };
    }

    // -- drawing ---------------------------------------------------------------
    draw() {
      if (!this.spec) return;
      const { w, h } = this._size();
      const m = this.margin, sp = this.spec;
      const { x, y } = this._scales(w, h);
      this.x = x; this.y = y; this.w = w; this.h = h;

      const ink = cssVar('--text-secondary'), grid = cssVar('--grid'), border = cssVar('--border-strong');
      this.svg.attr('width', w).attr('height', h);
      this.bg.attr('width', w).attr('height', h);

      // grid + axes
      const xt = x.ticks(Math.max(2, Math.round((w - m.left - m.right) / 110)));
      let yt;
      if (sp.yLog) {
        const all = y.ticks(Math.max(2, Math.round((h - m.top - m.bottom) / 45)));
        yt = all.filter((v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9);
        if (yt.length < 2) {     // zoomed inside a decade: use intermediate ticks
          const k = Math.ceil(all.length / 6);
          yt = all.filter((_, i) => i % k === 0);
        }
      } else yt = y.ticks(Math.max(2, Math.round((h - m.top - m.bottom) / 50)));
      this.gGrid.selectAll('line.gx').data(xt).join('line').attr('class', 'gx')
        .attr('x1', (d) => x(d)).attr('x2', (d) => x(d)).attr('y1', m.top).attr('y2', h - m.bottom).attr('stroke', grid);
      this.gGrid.selectAll('line.gy').data(yt).join('line').attr('class', 'gy')
        .attr('x1', m.left).attr('x2', w - m.right).attr('y1', (d) => y(d)).attr('y2', (d) => y(d)).attr('stroke', grid);

      const fx = tickFormatter(x, false), fy = tickFormatter(y, sp.yLog);
      this.gX.attr('transform', `translate(0,${h - m.bottom})`).call(d3.axisBottom(x).tickValues(xt).tickFormat(fx).tickSizeOuter(0).tickSize(4));
      this.gY.attr('transform', `translate(${m.left},0)`).call(d3.axisLeft(y).tickValues(yt).tickFormat(fy).tickSizeOuter(0).tickSize(4));
      this.svg.selectAll('.axis path, .axis line').attr('stroke', border);
      this.svg.selectAll('.axis text').attr('fill', ink);
      richAll(this.svg.selectAll('.axis .tick text'));
      // Make room for the widest y tick label next to the axis title.
      const need = Math.ceil(this.gY.node().getBBox().width) + 26;
      if (Math.abs(need - m.left) > 3 && !this._fitting) {
        m.left = Math.max(44, need); this._fitting = true; this.draw(); this._fitting = false; return;
      }
      rich(this.xTitle.attr('x', (m.left + w - m.right) / 2).attr('y', h - 6).attr('fill', ink), sp.xLabel || '');
      rich(this.yTitle.attr('transform', `translate(14,${(m.top + h - m.bottom) / 2}) rotate(-90)`).attr('fill', ink), sp.yLabel || '');

      // reference lines
      const refs = (sp.hlines || []).filter((r) => isFinite(r.y) && (!sp.yLog || r.y > 0));
      const [ylo, yhi] = y.domain();
      const inside = refs.filter((r) => r.y >= Math.min(ylo, yhi) && r.y <= Math.max(ylo, yhi));
      this.gRefs.selectAll('line').data(inside).join('line')
        .attr('x1', m.left).attr('x2', w - m.right).attr('y1', (r) => y(r.y)).attr('y2', (r) => y(r.y))
        .attr('stroke', (r) => r.color).attr('stroke-width', (r) => r.width || 1)
        .attr('stroke-dasharray', (r) => (DASH[r.dash || 'dot'] || []).join(' '));
      this.gRefs.selectAll('text').data(inside.filter((r) => r.label)).join('text')
        .attr('x', m.left + 4).attr('y', (r) => y(r.y) - 3).attr('fill', (r) => r.labelColor || ink)
        .attr('font-size', 10).each(function (r) { rich(d3.select(this), r.label); });

      this.overlay.attr('x', m.left).attr('y', m.top).attr('width', Math.max(0, w - m.left - m.right)).attr('height', Math.max(0, h - m.top - m.bottom));
      this.resetBtn.style('display', this.view || this.yView ? 'block' : 'none');
      this._project();
      this._paint();
    }

    // Pixel coordinates of every visible point, reused by painting and hover.
    _project() {
      const x = this.x, y = this.y, log = this.spec.yLog;
      this.proj = this._visibleSeries().map((s) => {
        const n = s.y.length, px = new Float32Array(n), py = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          const xv = s.x[i], yv = s.y[i];
          if (xv === null || yv === null || !isFinite(yv) || (log && yv <= 0)) { px[i] = NaN; py[i] = NaN; }
          else { px[i] = x(xv); py[i] = y(yv); }
        }
        return { s, px, py };
      });
    }

    _paint() {
      const dpr = window.devicePixelRatio || 1, { w, h } = this, m = this.margin;
      const c = this.canvas;
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
        c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
        c.style.width = w + 'px'; c.style.height = h + 'px';
      }
      const ctx = c.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.save();
      ctx.beginPath(); ctx.rect(m.left, m.top, w - m.left - m.right, h - m.top - m.bottom); ctx.clip();
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      const hov = this.hovered;
      const order = this.proj.slice().sort((a, b) => (a.s.name === hov) - (b.s.name === hov));
      for (const { s, px, py } of order) {
        const focus = hov === null || s.name === hov;
        ctx.globalAlpha = focus ? 1 : 0.22;
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.name === hov ? Math.max(3, 2 * (s.width || 1.4)) : (s.width || 1.4);
        ctx.setLineDash(DASH[s.dash || 'solid'] || []);
        ctx.beginPath();
        let pen = false;
        for (let i = 0; i < px.length; i++) {
          const X = px[i], Y = py[i];
          if (X !== X) { pen = false; continue; }   // NaN breaks the line
          if (pen) ctx.lineTo(X, Y); else { ctx.moveTo(X, Y); pen = true; }
        }
        ctx.stroke();
      }
      ctx.restore();
    }

    // -- legend ------------------------------------------------------------------
    _renderLegend() {
      if (!this.legend) return;
      const items = (this.spec.series || []).filter((s) => s.legend !== false);
      // Mode series named "(l, m)" are grouped in one scrollable column per l;
      // anything else (totals, references) gets a column of its own first.
      const lm = (s) => { const k = /^\((-?\d+),\s*(-?\d+)\)$/.exec(String(s.name).replace(/<[^>]+>/g, '')); return k ? [+k[1], +k[2]] : null; };
      const grouped = items.filter((s) => lm(s)).length >= 8;
      const sig = (grouped ? 'g|' : 'f|') + items.map((s) => s.name).join('\u0001');
      if (sig !== this._legendSig) {
        this._legendSig = sig;
        const L = this.legend.classed('grouped', grouped).html('');
        const groups = new Map();
        if (grouped) {
          items.forEach((s) => { const k = lm(s); const key = k ? k[0] : 'other'; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(s); });
          const keys = [...groups.keys()].sort((a, b) => (a === 'other' ? -1 : b === 'other' ? 1 : a - b));
          keys.forEach((key) => {
            const col = L.append('div').attr('class', 'lcol');
            col.append('button').attr('type', 'button').attr('class', 'lhead').datum(groups.get(key))
              .attr('title', key === 'other' ? '' : 'Click: show or hide all of l = ' + key + ' · double-click: only these')
              .html(key === 'other' ? 'other' : `<i>l</i> = ${key}<span>${groups.get(key).length}</span>`);
            const list = col.append('div').attr('class', 'llist');
            groups.get(key).forEach((s) => list.append('button').attr('type', 'button').attr('class', 'li').datum(s));
          });
        } else {
          items.forEach((s) => L.append('button').attr('type', 'button').attr('class', 'li').datum(s));
        }
        const all = L.selectAll('button.li');
        all.append('svg').attr('width', 18).attr('height', 8).append('line').attr('x1', 1).attr('x2', 17).attr('y1', 4).attr('y2', 4);
        all.append('span').html((s) => { const k = grouped && lm(s); return k ? `<i>m</i> = ${k[1] < 0 ? '−' + -k[1] : k[1]}` : s.name; });
        all.attr('title', (s) => String(s.name).replace(/<[^>]+>/g, ''));
        let clickTimer = null;
        const toggle = (names) => {
          const allHidden = names.every((n) => this.hidden.has(n));
          names.forEach((n) => (allHidden ? this.hidden.delete(n) : this.hidden.add(n)));
          this._renderLegend(); this.draw();
        };
        const isolate = (names) => {
          const every = items.map((d) => d.name);
          const isolated = every.every((n) => names.includes(n) || this.hidden.has(n)) && names.every((n) => !this.hidden.has(n));
          this.hidden = isolated ? new Set() : new Set(every.filter((n) => !names.includes(n)));
          this._renderLegend(); this.draw();
        };
        const clicks = (sel, names) => sel.on('click', (ev, d) => {
          clearTimeout(clickTimer); clickTimer = setTimeout(() => toggle(names(d)), 220);
        }).on('dblclick', (ev, d) => { clearTimeout(clickTimer); isolate(names(d)); });
        clicks(all, (s) => [s.name]);
        clicks(L.selectAll('button.lhead'), (list) => list.map((s) => s.name));
        all.on('mouseenter', (ev, s) => { if (!this.hidden.has(s.name)) { this.hovered = s.name; this._paint(); } })
          .on('mouseleave', () => { this.hovered = null; this._paint(); });
      }
      const all = this.legend.selectAll('button.li');
      all.classed('off', (s) => this.hidden.has(s.name));
      all.select('line').attr('stroke', (s) => s.color).attr('stroke-width', (s) => Math.max(1.6, s.width || 1.4))
        .attr('stroke-dasharray', (s) => (DASH[s.dash || 'solid'] || []).join(' '));
      this.legend.selectAll('button.lhead').classed('off', (list) => list.every((s) => this.hidden.has(s.name)));
    }

    // -- export ---------------------------------------------------------------------
    // The plot as a standalone SVG: grid, curves as paths (points closer than
    // a third of a pixel are merged), axes, and the legend of the visible
    // series to the right, one column per l when the modes are grouped.
    toSVG() {
      const { w, h } = this, m = this.margin;
      const pw = w - m.left - m.right, ph = h - m.top - m.bottom;
      const id = 'c' + Math.random().toString(36).slice(2);
      let body = cloneLayer(this.bg.node());
      body += `<defs><clipPath id="${id}"><rect x="${r2(m.left)}" y="${r2(m.top)}" width="${r2(pw)}" height="${r2(ph)}"/></clipPath></defs><g clip-path="url(#${id})" fill="none" stroke-linejoin="round" stroke-linecap="round">`;
      for (const { s, px, py } of this.proj) {
        let d = '', pen = false, lx = 0, ly = 0;
        for (let i = 0; i < px.length; i++) {
          const X = px[i], Y = py[i];
          if (X !== X) { pen = false; continue; }
          const last = i === px.length - 1 || px[i + 1] !== px[i + 1];
          if (!pen) { d += `M${r2(X)} ${r2(Y)}`; pen = true; lx = X; ly = Y; }
          else if (last || Math.abs(X - lx) > 0.33 || Math.abs(Y - ly) > 0.33) { d += `L${r2(X)} ${r2(Y)}`; lx = X; ly = Y; }
        }
        if (!d) continue;
        const dash = (DASH[s.dash || 'solid'] || []).join(' ');
        body += `<path d="${d}" stroke="${s.color}" stroke-width="${s.width || 1.4}"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
      }
      body += '</g>' + cloneLayer(this.svg.node());
      const leg = this._legendSVG(w + 4, m.top, ph);
      return { svg: svgDoc(w + leg.width, h, body + leg.body), w: w + leg.width, h };
    }

    _legendSVG(x0, y0, maxH) {
      if (!this.legend) return { body: '', width: 0 };
      const items = (this.spec.series || []).filter((s) => s.legend !== false && !this.hidden.has(s.name) && s.y && s.y.length);
      if (!items.length) return { body: '', width: 0 };
      const lm = (s) => { const k = /^\((-?\d+),\s*(-?\d+)\)$/.exec(plain(s.name)); return k ? [+k[1], +k[2]] : null; };
      const grouped = (this.spec.series || []).filter((s) => s.legend !== false && lm(s)).length >= 8;
      const ink = cssVar('--text-secondary');
      const ROW = 14, rows = Math.max(3, Math.floor(maxH / ROW));
      // Columns of {head, entries}; a long group continues in the next column.
      const cols = [];
      const push = (head, list, label) => {
        for (let i = 0; i < list.length; i += rows - (head ? 1 : 0)) {
          const part = list.slice(i, i + rows - (head ? 1 : 0));
          cols.push({ head: i === 0 ? head : head && '', entries: part.map((s) => ({ s, label: label(s) })) });
        }
      };
      if (grouped) {
        const groups = new Map();
        items.forEach((s) => { const k = lm(s); const key = k ? k[0] : 'other'; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(s); });
        [...groups.keys()].sort((a, b) => (a === 'other' ? -1 : b === 'other' ? 1 : a - b)).forEach((key) =>
          push(key === 'other' ? '' : { l: key }, groups.get(key), (s) => { const k = lm(s); return k ? { m: k[1] } : { text: s.name }; }));
      } else push('', items, (s) => ({ text: s.name }));
      let body = '', x = x0 + 8;
      for (const col of cols) {
        let y = y0 + 10, cw = 0;
        if (col.head !== '') {
          if (col.head) body += `<text x="${r2(x)}" y="${r2(y)}" font-size="11px" font-weight="bold" fill="${ink}"><tspan font-style="italic">l</tspan>\u00a0=\u00a0${col.head.l}</text>`;
          cw = Math.max(cw, textWidth('l = ' + (col.head ? col.head.l : ''), 11, 600));
          y += ROW;
        }
        for (const { s, label } of col.entries) {
          const dash = (DASH[s.dash || 'solid'] || []).join(' ');
          body += `<line x1="${r2(x)}" x2="${r2(x + 18)}" y1="${r2(y - 4)}" y2="${r2(y - 4)}" stroke="${s.color}" stroke-width="${Math.max(1.6, s.width || 1.4)}" stroke-linecap="round"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
          const tx = x + 24;
          if (label.m !== undefined) {
            body += `<text x="${r2(tx)}" y="${r2(y)}" font-size="11px" fill="${ink}"><tspan font-style="italic">m</tspan>\u00a0=\u00a0${label.m < 0 ? '−' + -label.m : label.m}</text>`;
            cw = Math.max(cw, 24 + textWidth('m = −0', 11));
          } else {
            const t = document.createElementNS(NS, 'text');
            rich(d3.select(t), label.text);
            t.querySelectorAll('tspan').forEach((ts) => {
              if (ts.getAttribute('font-size')) ts.setAttribute('font-size', '8.25px');
              const dy = ts.getAttribute('dy'); if (dy) ts.setAttribute('dy', r2(parseFloat(dy) * 11) + '');
            });
            body += `<text x="${r2(tx)}" y="${r2(y)}" font-size="11px" fill="${ink}">${t.innerHTML.replace(/ xmlns="[^"]*"/g, '')}</text>`;
            cw = Math.max(cw, 24 + textWidth(label.text, 11));
          }
          y += ROW;
        }
        x += cw + 16;
      }
      return { body, width: x - x0 + 4 };
    }

    // -- pointer: hover tooltip and brush zoom -------------------------------------
    resetZoom() {
      if (!this.view && !this.yView) return;
      this.view = null; this.yView = null; this._hideTip(); this.draw();
    }

    _plotRect() {
      const m = this.margin;
      return { l: m.left, r: this.w - m.right, t: m.top, b: this.h - m.bottom };
    }

    // Current domains as [lo, hi], whatever the axis direction.
    _domains() {
      const xd = this.x.domain(), yd = this.y.domain();
      return { xd: [Math.min(...xd), Math.max(...xd)], yd: [Math.min(...yd), Math.max(...yd)] };
    }

    // Drag: a box zooms both axes; a nearly horizontal drag zooms x only and
    // a nearly vertical one y only. Shift-drag pans. Ctrl/⌘ + wheel (or a
    // trackpad pinch) zooms around the cursor. Double-click resets.
    _bindPointer() {
      let start = null, mode = null, pan = null;
      const ov = this.overlay, SNAP = 10;
      const clampPt = ([mx, my]) => {
        const r = this._plotRect();
        return [Math.max(r.l, Math.min(r.r, mx)), Math.max(r.t, Math.min(r.b, my))];
      };
      ov.on('pointerdown', (ev) => {
        if (ev.button !== 0) return;
        ev.preventDefault();   // no text selection while dragging
        start = clampPt(d3.pointer(ev, this.svg.node()));
        mode = (this.dragMode === 'pan') !== ev.shiftKey ? 'pan' : 'box';
        if (mode === 'pan') pan = { x: this.x.copy(), y: this.y.copy() };
        ov.node().setPointerCapture(ev.pointerId);
        ov.style('cursor', mode === 'pan' ? 'grabbing' : null);
      });
      ov.on('pointermove', (ev) => {
        const raw = d3.pointer(ev, this.svg.node());
        if (start === null) { this._hover(raw[0], raw[1]); return; }
        this._hideTip();
        if (mode === 'pan') {
          const dx = raw[0] - start[0], dy = raw[1] - start[1], r = this._plotRect();
          const nx = [pan.x.invert(r.l - dx), pan.x.invert(r.r - dx)];
          const ny = [pan.y.invert(r.b - dy), pan.y.invert(r.t - dy)];
          this.view = [Math.min(...nx), Math.max(...nx)];
          this.yView = [Math.min(...ny), Math.max(...ny)];
          this.draw();
          return;
        }
        const [mx, my] = clampPt(raw), r = this._plotRect();
        const w = Math.abs(mx - start[0]), h = Math.abs(my - start[1]);
        let x0 = Math.min(start[0], mx), x1 = Math.max(start[0], mx);
        let y0 = Math.min(start[1], my), y1 = Math.max(start[1], my);
        if (h < SNAP && w >= SNAP) { y0 = r.t; y1 = r.b; }        // x only
        else if (w < SNAP && h >= SNAP) { x0 = r.l; x1 = r.r; }   // y only
        this.brush.attr('visibility', 'visible').attr('x', x0).attr('width', x1 - x0).attr('y', y0).attr('height', y1 - y0);
      });
      ov.on('pointerup', (ev) => {
        if (start === null) return;
        ov.style('cursor', null);
        if (mode === 'box') {
          const [mx, my] = clampPt(d3.pointer(ev, this.svg.node()));
          this.brush.attr('visibility', 'hidden');
          const w = Math.abs(mx - start[0]), h = Math.abs(my - start[1]);
          if (w >= SNAP || h >= SNAP) {
            if (w >= SNAP) { const a = this.x.invert(start[0]), b = this.x.invert(mx); this.view = [Math.min(a, b), Math.max(a, b)]; }
            if (h >= SNAP) { const a = this.y.invert(start[1]), b = this.y.invert(my); this.yView = [Math.min(a, b), Math.max(a, b)]; }
            this._hideTip();
            this.draw();
          }
        }
        start = null; mode = null; pan = null;
      });
      ov.on('pointerleave', () => { if (start === null) this._hideTip(); });
      ov.on('dblclick', () => this.resetZoom());
      ov.on('wheel', (ev) => {
        if (!(ev.ctrlKey || ev.metaKey)) return;    // plain wheel scrolls the page
        ev.preventDefault();
        const [mx, my] = d3.pointer(ev, this.svg.node());
        const k = Math.exp(Math.max(-0.5, Math.min(0.5, ev.deltaY * 0.01)));   // >1 zooms out
        const r = this._plotRect();
        const zx = (s, p, lo, hi) => [s.invert(p + (lo - p) * k), s.invert(p + (hi - p) * k)];
        const nx = zx(this.x, mx, r.l, r.r), ny = zx(this.y, my, r.b, r.t);
        this.view = [Math.min(...nx), Math.max(...nx)];
        this.yView = [Math.min(...ny), Math.max(...ny)];
        this._hideTip();
        this.draw();
      }, { passive: false });
    }

    _hideTip() {
      this.tip.classed('show', false);
      this.gHover.selectAll('*').remove();
      if (this.hovered !== null) { this.hovered = null; this._paint(); }
    }

    _hover(mx, my) {
      if (!this.proj || !this.proj.length) return;
      const sp = this.spec, fx = sp.xFormat || ((v) => fmtNum(v, 6)), fy = sp.yFormat || ((v) => fmtNum(v, 4));
      const surface = cssVar('--surface-1');
      if (this.opts.hover === 'closest') {
        // Nearest point of any visible curve, in pixels.
        let best = null, bd = 900;   // 30 px radius
        for (const p of this.proj) {
          const { px, py } = p;
          for (let i = 0; i < px.length; i++) {
            if (py[i] < this.margin.top || py[i] > this.h - this.margin.bottom) continue;
            const dx = px[i] - mx, dy = py[i] - my, d = dx * dx + dy * dy;
            if (d < bd) { bd = d; best = { p, i }; }
          }
        }
        if (!best) { this._hideTip(); return; }
        const { p, i } = best;
        if (this.hovered !== p.s.name) { this.hovered = p.s.name; this._paint(); }
        this.gHover.selectAll('circle').data([0]).join('circle').attr('r', 4.5)
          .attr('cx', p.px[i]).attr('cy', p.py[i]).attr('fill', p.s.color).attr('stroke', surface).attr('stroke-width', 2);
        this.gHover.selectAll('line').remove();
        this._showTip(`<b><i style="background:${p.s.color}"></i>${p.s.name}</b><div>${sp.xLabel || 'x'} = ${fx(p.s.x[i])}</div><div>${fy(p.s.y[i])}</div>`, p.px[i], p.py[i]);
      } else {
        // x-unified: the sample nearest in x, every visible series at that index.
        const ref = this.proj[0];
        let bi = -1, bd = Infinity;
        for (let i = 0; i < ref.px.length; i++) {
          const d = Math.abs(ref.px[i] - mx);
          if (d < bd) { bd = d; bi = i; }
        }
        if (bi < 0) { this._hideTip(); return; }
        const X = ref.px[bi];
        this.gHover.selectAll('line').data([0]).join('line').attr('x1', X).attr('x2', X)
          .attr('y1', this.margin.top).attr('y2', this.h - this.margin.bottom).attr('stroke', cssVar('--text-muted')).attr('stroke-dasharray', '3 3');
        const pts = this.proj.filter((p) => p.py[bi] >= this.margin.top - 0.5 && p.py[bi] <= this.h - this.margin.bottom + 0.5);
        this.gHover.selectAll('circle').data(pts).join('circle').attr('r', 3.5)
          .attr('cx', X).attr('cy', (p) => p.py[bi]).attr('fill', (p) => p.s.color).attr('stroke', surface).attr('stroke-width', 1.5);
        const rows = pts.map((p) => `<div><i style="background:${p.s.color}"></i>${p.s.name}<span>${fy(p.s.y[bi])}</span></div>`).join('');
        this._showTip(`<b>${sp.xLabel || 'x'} = ${fx(ref.s.x[bi])}</b>${rows}`, X, my);
      }
    }

    _showTip(html, px, py) {
      const t = this.tip.html(html).classed('show', true).node();
      const tw = t.offsetWidth, th = t.offsetHeight;
      let left = px + 14, top = py - th / 2;
      if (left + tw > this.w - 4) left = px - tw - 14;
      top = Math.max(4, Math.min(this.h - th - 4, top));
      t.style.transform = `translate(${left}px, ${top}px)`;
    }
  }

  // ---------------------------------------------------------------------------
  // Heatmap with equal axes, colour bar and point markers
  // ---------------------------------------------------------------------------
  class HeatmapChart {
    constructor(el) {
      this.el = el;
      el.classList.add('d3chart');
      el.innerHTML = '';
      this.area = d3.select(el).append('div').attr('class', 'd3area');
      this.canvas = this.area.append('canvas').node();
      this.svg = this.area.append('svg');
      this.title = this.svg.append('text').attr('class', 'chart-title');
      this.gX = this.svg.append('g').attr('class', 'axis x');
      this.gY = this.svg.append('g').attr('class', 'axis y');
      this.xTitle = this.svg.append('text').attr('class', 'axis-title').attr('text-anchor', 'middle');
      this.yTitle = this.svg.append('text').attr('class', 'axis-title').attr('text-anchor', 'middle');
      this.defs = this.svg.append('defs');
      this.gBar = this.svg.append('g').attr('class', 'axis bar');
      this.gCross = this.svg.append('g').attr('class', 'origin');
      this.gMarks = this.svg.append('g');
      this.overlay = this.svg.append('rect').attr('class', 'overlay');
      this.tip = this.area.append('div').attr('class', 'd3tip');
      addExport(this.area.append('div').attr('class', 'd3tools'), this);
      this.gradId = 'g' + Math.random().toString(36).slice(2);
      this._ro = new ResizeObserver(() => this.draw());
      this._ro.observe(el);
      this.overlay.on('pointermove', (ev) => this._hover(ev)).on('pointerleave', () => this.tip.classed('show', false));
    }

    update(spec) { this.spec = spec; this._image(); this.draw(); }

    // Paint the grid once into an offscreen canvas; draw() only scales it.
    _image() {
      // z: rows (arrays or Float32Array views), row 0 at y = min; null/NaN is transparent.
      // spec.zDomain = [lo, hi] fixes the colour scale (values outside are clamped).
      const { z, ramp } = this.spec;
      const n = z.length, mcols = z[0].length;
      let lo = Infinity, hi = -Infinity;
      if (this.spec.zDomain) [lo, hi] = this.spec.zDomain;
      else for (const row of z) for (let i = 0; i < row.length; i++) { const v = row[i]; if (v !== null && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
      if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
      if (lo === hi) hi = lo + 1;
      this.zr = [lo, hi];
      const color = d3.scaleLinear().domain(ramp.map((r) => lo + r[0] * (hi - lo))).range(ramp.map((r) => r[1])).interpolate(d3.interpolateRgb).clamp(true);
      // 1024-entry lookup table instead of a d3 colour per pixel.
      const N = 1024, lut = new Uint8ClampedArray(3 * N);
      for (let k = 0; k < N; k++) { const c = d3.rgb(color(lo + (hi - lo) * k / (N - 1))); lut[3 * k] = c.r; lut[3 * k + 1] = c.g; lut[3 * k + 2] = c.b; }
      const scale = (N - 1) / (hi - lo);
      const off = this.off && this.off.width === mcols && this.off.height === n ? this.off : document.createElement('canvas');
      off.width = mcols; off.height = n;
      const ctx = off.getContext('2d'), img = ctx.createImageData(mcols, n), px = img.data;
      for (let j = 0; j < n; j++) {
        const row = z[n - 1 - j];  // row 0 is y = min, drawn at the bottom
        for (let i = 0, k = 4 * j * mcols; i < mcols; i++, k += 4) {
          const v = row[i];
          if (v === null || !(v === v) || v === Infinity || v === -Infinity) { px[k + 3] = 0; continue; }
          let t = Math.round((v - lo) * scale); t = t < 0 ? 0 : t >= N ? N - 1 : t;
          px[k] = lut[3 * t]; px[k + 1] = lut[3 * t + 1]; px[k + 2] = lut[3 * t + 2]; px[k + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      this.off = off;
      this.color = color;
    }

    draw() {
      if (!this.spec) return;
      const sp = this.spec, r = this.area.node().getBoundingClientRect();
      const w = Math.max(80, r.width), h = Math.max(80, r.height);
      const m = { top: sp.title ? 26 : 12, right: 74, bottom: 40, left: 56 };
      const side = Math.max(40, Math.min(w - m.left - m.right, h - m.top - m.bottom));
      const ox = m.left + (w - m.left - m.right - side) / 2, oy = m.top;
      const [a, b] = sp.extent;
      const x = d3.scaleLinear().domain([a, b]).range([ox, ox + side]);
      const y = d3.scaleLinear().domain([a, b]).range([oy + side, oy]);
      this.x = x; this.y = y; this.box = [ox, oy, side]; this.w = w; this.h = h;
      const ink = cssVar('--text-secondary'), border = cssVar('--border-strong');

      const dpr = window.devicePixelRatio || 1, c = this.canvas;
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.width = w + 'px'; c.style.height = h + 'px';
      const ctx = c.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.off, ox, oy, side, side);

      this.svg.attr('width', w).attr('height', h);
      rich(this.title.attr('x', m.left).attr('y', 14).attr('fill', ink), sp.title || '');
      const nt = Math.max(3, Math.round(side / 90));
      this.gX.attr('transform', `translate(0,${oy + side})`).call(d3.axisBottom(x).ticks(nt).tickFormat(d3.format('~g')).tickSize(4).tickSizeOuter(0));
      this.gY.attr('transform', `translate(${ox},0)`).call(d3.axisLeft(y).ticks(nt).tickFormat(d3.format('~g')).tickSize(4).tickSizeOuter(0));
      rich(this.xTitle.attr('x', ox + side / 2).attr('y', oy + side + 34).attr('fill', ink), sp.xLabel || '');
      rich(this.yTitle.attr('transform', `translate(${ox - 40},${oy + side / 2}) rotate(-90)`).attr('fill', ink), sp.yLabel || '');

      // colour bar
      const grad = this.defs.selectAll('linearGradient').data([0]).join('linearGradient').attr('id', this.gradId)
        .attr('x1', 0).attr('x2', 0).attr('y1', 1).attr('y2', 0);
      grad.selectAll('stop').data(sp.ramp).join('stop').attr('offset', (d) => d[0]).attr('stop-color', (d) => d[1]);
      const bx = ox + side + 18, bar = d3.scaleLinear().domain(this.zr).range([oy + side, oy]);
      this.gBar.selectAll('rect').data([0]).join('rect').attr('x', bx).attr('y', oy).attr('width', 10).attr('height', side)
        .attr('rx', 3).attr('fill', `url(#${this.gradId})`);
      this.gBar.selectAll('g.ax').data([0]).join('g').attr('class', 'ax').attr('transform', `translate(${bx + 10},0)`)
        .call(d3.axisRight(bar).ticks(5).tickFormat((v) => fmtNum(v, 2)).tickSize(3));
      richAll(this.svg.selectAll('.tick text'));
      // The bar's label sits clear of its widest tick label.
      const lx = Math.max(w - 6, bx + 10 + Math.ceil(this.gBar.select('g.ax').node().getBBox().width) + 16);
      this.gBar.selectAll('text.lbl').data([0]).join('text').attr('class', 'lbl axis-title')
        .attr('transform', `translate(${lx},${oy + side / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('fill', ink).call((t) => rich(t, sp.zLabel || ''));
      this.svg.selectAll('.axis path, .axis line').attr('stroke', border);
      this.svg.selectAll('.axis text').attr('fill', ink);

      // dashed lines through the origin (the centre of mass)
      const origin = sp.origin !== false && a <= 0 && b >= 0 ? [
        { x1: x(0), x2: x(0), y1: oy, y2: oy + side },
        { x1: ox, x2: ox + side, y1: y(0), y2: y(0) }] : [];
      this.gCross.selectAll('line').data(origin).join('line')
        .attr('x1', (d) => d.x1).attr('x2', (d) => d.x2).attr('y1', (d) => d.y1).attr('y2', (d) => d.y2)
        .attr('stroke', cssVar('--text-secondary')).attr('stroke-opacity', 0.7).attr('stroke-width', 1)
        .attr('stroke-dasharray', '5 4');

      // markers
      const surface = cssVar('--surface-1');
      this.gMarks.selectAll('circle').data(sp.markers || []).join('circle')
        .attr('cx', (d) => x(d.x)).attr('cy', (d) => y(d.y)).attr('r', 5.5)
        .attr('fill', (d) => (d.filled ? d.color : 'none')).attr('stroke', (d) => (d.filled ? '#fff' : d.color))
        .attr('stroke-width', (d) => (d.filled ? 2 : 2.4))
        .attr('visibility', (d) => (d.x >= a && d.x <= b && d.y >= a && d.y <= b ? 'visible' : 'hidden'));
      this.overlay.attr('x', ox).attr('y', oy).attr('width', side).attr('height', side);
    }

    // The slice as a standalone SVG: the field as an embedded image (resampled
    // smoothly, as on screen), with the axes, colour bar and markers as vectors.
    toSVG() {
      const { w, h } = this, [ox, oy, side] = this.box;
      const k = Math.min(1600, Math.max(this.off.width, Math.round(2 * side)));
      const up = document.createElement('canvas'); up.width = k; up.height = k;
      const ctx = up.getContext('2d'); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(this.off, 0, 0, k, k);
      const img = `<image x="${r2(ox)}" y="${r2(oy)}" width="${r2(side)}" height="${r2(side)}" preserveAspectRatio="none" xlink:href="${up.toDataURL('image/png')}"/>`;
      const bb = this.svg.node().getBBox(), W = Math.max(w, Math.ceil(bb.x + bb.width) + 6);
      return { svg: svgDoc(W, h, img + cloneLayer(this.svg.node())), w: W, h };
    }

    _hover(ev) {
      const [mx, my] = d3.pointer(ev, this.svg.node());
      const sp = this.spec, xv = this.x.invert(mx), yv = this.y.invert(my);
      const n = sp.z.length, mcols = sp.z[0].length, [a, b] = sp.extent;
      const i = Math.round((xv - a) / (b - a) * (mcols - 1)), j = Math.round((yv - a) / (b - a) * (n - 1));
      if (i < 0 || j < 0 || i >= mcols || j >= n) { this.tip.classed('show', false); return; }
      const near = (sp.markers || []).find((d) => Math.hypot(this.x(d.x) - mx, this.y(d.y) - my) < 9);
      const html = near ? `<b>${near.name}</b><div>x = ${fmtNum(near.x, 4)}, y = ${fmtNum(near.y, 4)}</div>`
                        : `<div>x = ${fmtNum(xv, 4)}, y = ${fmtNum(yv, 4)}</div><b>${fmtNum(sp.z[j][i], 4)}</b>`;
      const t = this.tip.html(html).classed('show', true).node();
      let left = mx + 14; if (left + t.offsetWidth > this.w - 4) left = mx - t.offsetWidth - 14;
      t.style.transform = `translate(${left}px, ${Math.max(4, my - t.offsetHeight / 2)}px)`;
    }
  }

  const D3Charts = window.D3Charts = { LineChart, HeatmapChart, fmtNum, exportName: null, onError: null };
})();
