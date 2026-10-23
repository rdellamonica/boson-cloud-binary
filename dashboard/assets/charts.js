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
      this.gBar.selectAll('text.lbl').data([0]).join('text').attr('class', 'lbl axis-title')
        .attr('transform', `translate(${w - 6},${oy + side / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('fill', ink).call((t) => rich(t, sp.zLabel || ''));
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

  window.D3Charts = { LineChart, HeatmapChart, fmtNum };
})();
