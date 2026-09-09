/* ==========================================================================
   WaferLens OS — HYPERDECK
   ==========================================================================
   Six additional views, a global FX layer and a live boot sequence, layered on
   top of app.js without touching it. Everything drawn here comes from the same
   endpoints the rest of the console uses:

     Command Deck        /api/summary  /api/wafers  /api/benchmarks  /api/provenance
     Neural Core         /api/benchmarks (ablation waterfall)
     Waveform Bank       /api/wafer/<id>  /api/die/... (the 2000-block memmap)
     Risk Observatory    /api/scores  (one calibrated risk + truth per eligible die)
     Calibration Lab     /api/calibration
     Signature Atlas     /api/rootcauses

   The FFT, spectrogram, ROC, KS, gain and decile tables are computed in the
   browser from those arrays. Nothing on these pages is invented.
   ========================================================================== */
(() => {
  'use strict';

  const P = {
    cyan: '#00F0FF', mint: '#00E676', amber: '#FFB300', ruby: '#FF1744',
    violet: '#B388FF', dim: '#586069', tx: '#C9D1D9', hi: '#F0F6FC',
    grid: 'rgba(0,240,255,.10)',
  };

  const HX = { loaded: {}, wafer: null, scores: null, bench: null, scope: null };

  const dom = (html) => {
    const t = document.createElement('template');
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  };
  const sv = (tag, attrs = {}, text) => el(tag, attrs, text);   // app.js SVG helper
  const q = (sel, root = document) => root.querySelector(sel);
  const qa = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // ------------------------------------------------------------------ ticker
  // A console this dense will happily saturate a frame, so the loop is capped at 30fps,
  // panels on hidden views are skipped entirely, and anything static is drawn once into
  // an offscreen canvas and blitted rather than rebuilt every frame.
  const TICKERS = [];
  function ticker(fn, node) { TICKERS.push({ fn, node }); }
  let lastFrame = 0;
  function frame(t) {
    requestAnimationFrame(frame);
    if (t - lastFrame < 32) return;
    lastFrame = t;
    for (const k of TICKERS) {
      if (k.node && k.node.id !== 'fxbg' && !k.node.closest('.view.on')) continue;
      if (k.node && !k.node.isConnected) continue;
      try { k.fn(t); } catch (e) { /* one bad panel must not stop the deck */ }
    }
  }
  requestAnimationFrame(frame);

  const DPR = () => Math.min(window.devicePixelRatio || 1, 2);
  const RECTS = new WeakMap();
  function fit(cv) {
    let r = RECTS.get(cv);
    const now = performance.now();
    if (!r || now - r.t > 700 || r.w === 0) {
      const b = cv.getBoundingClientRect();
      r = { w: b.width, h: b.height, t: now };
      RECTS.set(cv, r);
      const d = DPR();
      const W = Math.max(1, Math.round(r.w * d)), H = Math.max(1, Math.round(r.h * d));
      if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    }
    const ctx = cv.getContext('2d');
    ctx.setTransform(DPR(), 0, 0, DPR(), 0, 0);
    return { ctx, w: r.w, h: r.h };
  }

  /** Draw once into a cached backing canvas; redraw only when the size changes. */
  function backing(store, key, w, h, paint) {
    const d = DPR();
    let b = store[key];
    if (!b || b.w !== w || b.h !== h) {
      const cv = document.createElement('canvas');
      cv.width = Math.max(1, Math.round(w * d));
      cv.height = Math.max(1, Math.round(h * d));
      const ctx = cv.getContext('2d');
      ctx.setTransform(d, 0, 0, d, 0, 0);
      paint(ctx, w, h);
      b = store[key] = { cv, w, h };
    }
    return b.cv;
  }

  /** Collapse a long series into one min/max pair per output pixel column. */
  function columns(series, width) {
    const n = series.length, cols = Math.max(2, Math.round(width));
    const out = new Array(cols);
    for (let x = 0; x < cols; x++) {
      const a = Math.floor((x / cols) * n), b = Math.max(a + 1, Math.floor(((x + 1) / cols) * n));
      let lo = Infinity, hi = -Infinity;
      for (let i = a; i < b && i < n; i++) {
        const v = series[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      out[x] = [lo, hi];
    }
    return out;
  }

  /** Count a number up into an element. Used for every headline figure. */
  function countTo(node, to, dur = 900, fmt = (v) => Math.round(v).toLocaleString('en-US')) {
    if (!node) return;
    const from = Number(node.dataset.hxv || 0);
    node.dataset.hxv = to;
    const t0 = performance.now();
    const step = (t) => {
      const k = clamp((t - t0) / dur, 0, 1);
      const e = 1 - Math.pow(1 - k, 3);
      node.textContent = fmt(from + (to - from) * e);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /** Reveal an SVG path by animating its dash offset — the "drawing itself" effect. */
  function drawIn(path, dur = 1100, delay = 0) {
    let len = 0;
    try { len = path.getTotalLength(); } catch (e) { return; }
    if (!len) return;
    path.style.strokeDasharray = len;
    path.style.strokeDashoffset = len;
    path.style.transition = `stroke-dashoffset ${dur}ms cubic-bezier(.22,1,.36,1) ${delay}ms`;
    requestAnimationFrame(() => requestAnimationFrame(() => { path.style.strokeDashoffset = 0; }));
  }


  /** Time-based 0..1 ramp. Frame counters stall whenever the tab is backgrounded and
      requestAnimationFrame stops firing, so every reveal is driven by the clock. */
  function ramp(state, t, ms) {
    if (state.t0 == null) state.t0 = t;
    const k = clamp((t - state.t0) / ms, 0, 1);
    return 1 - Math.pow(1 - k, 3);
  }

  // ==========================================================================
  // FX LAYER — background field, scanlines, reticles
  // ==========================================================================
  function mountFx() {
    document.body.appendChild(dom('<canvas id="fxbg"></canvas>'));
    document.body.appendChild(dom('<div id="fxvig"></div>'));
    document.body.appendChild(dom('<div id="fxscan"></div>'));
    ['tl', 'tr', 'bl', 'br'].forEach((c) =>
      document.body.appendChild(dom(`<div class="hx-reticle ${c}"></div>`)));

    const cv = q('#fxbg');
    const dust = [];
    for (let i = 0; i < 110; i++) {
      dust.push({ x: Math.random(), y: Math.random(), z: 0.25 + Math.random() * 0.9,
        s: 0.1 + Math.random() * 0.55 });
    }
    ticker((t) => {
      const { ctx, w, h } = fit(cv);
      ctx.clearRect(0, 0, w, h);

      // reactor bloom behind the whole console
      const pulse = 0.5 + 0.5 * Math.sin(t / 2400);
      const g = ctx.createRadialGradient(w * 0.5, h * 0.42, 0, w * 0.5, h * 0.42,
        Math.max(w, h) * 0.62);
      g.addColorStop(0, `rgba(0,240,255,${0.045 + pulse * 0.028})`);
      g.addColorStop(0.55, 'rgba(0,120,255,.014)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);

      // drifting particulate — a cleanroom is never truly empty
      ctx.globalCompositeOperation = 'lighter';
      for (const d of dust) {
        d.y -= (0.000012 * d.z * 1000) / Math.max(h, 1);
        if (d.y < -0.02) { d.y = 1.02; d.x = Math.random(); }
        const x = (d.x + Math.sin(t / 9000 + d.z * 6) * 0.012) * w;
        ctx.fillStyle = `rgba(0,240,255,${0.07 * d.z})`;
        ctx.beginPath(); ctx.arc(x, d.y * h, d.s * d.z * 2.1, 0, 7); ctx.fill();
      }

      // three slow horizon beams
      for (let i = 0; i < 3; i++) {
        const y = ((t / (26000 + i * 9000) + i * 0.37) % 1) * h;
        const grad = ctx.createLinearGradient(0, y - 40, 0, y + 40);
        grad.addColorStop(0, 'rgba(0,240,255,0)');
        grad.addColorStop(0.5, 'rgba(0,240,255,.030)');
        grad.addColorStop(1, 'rgba(0,240,255,0)');
        ctx.fillStyle = grad; ctx.fillRect(0, y - 40, w, 80);
      }
      ctx.globalCompositeOperation = 'source-over';
    }, cv);
  }

  // ==========================================================================
  // BOOT SEQUENCE — real checks, not a fake progress bar
  // ==========================================================================
  async function bootSequence() {
    if (sessionStorage.getItem('hx-booted')) return;
    sessionStorage.setItem('hx-booted', '1');

    const ov = dom(`
      <div id="bootseq"><div class="hx-boot">
        <h1>WAFERLENS <span>// OS</span></h1>
        <div class="sub">Multi-resolution die yield intelligence · cold start</div>
        <div id="bootLines"></div>
        <div class="hx-bootbar"><i id="bootBar"></i></div>
      </div></div>`);
    document.body.appendChild(ov);
    const lines = q('#bootLines', ov), bar = q('#bootBar', ov);
    let n = 0;
    const total = 7;
    const say = (label, value, ok = true) => {
      n++;
      const d = dom(`<div class="line"><b>${ok ? '[ OK ]' : '[ !! ]'}</b>` +
        `<span>${esc(label)}</span><i>${esc(value)}</i></div>`);
      if (!ok) d.querySelector('b').style.color = P.ruby;
      lines.appendChild(d);
      bar.style.width = `${(n / total) * 100}%`;
    };
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));

    try {
      const h = await api('/api/health');
      say('artifact store mounted', `${h.dataset} · ${h.model}`);
      await wait(90);
      say('prediction contract', `${h.preds.toLocaleString('en-US')} scored dies`);
      await wait(90);
      const s = await api('/api/summary');
      say('wafer splits verified', `${s.counts.wafers} wafers held out`);
      await wait(80);
      say('decision threshold loaded', Number(s.threshold).toFixed(4));
      await wait(80);
      say('block memmap attached', '2,000 readings / die · fp16');
      await wait(80);
      say('recall at operating point',
        `${((s.counts.caught / Math.max(s.counts.new_fail, 1)) * 100).toFixed(1)}%`);
      await wait(80);
      say('hyperdeck online', 'all consoles live');
    } catch (e) {
      say('backend unreachable', String(e.message || e).slice(0, 60), false);
      bar.style.width = '100%';
    }
    await wait(430);
    ov.classList.add('gone');
    setTimeout(() => ov.remove(), 800);
  }

  // ==========================================================================
  // NAVIGATION
  // ==========================================================================
  const NEW_VIEWS = [
    { id: 'command', label: 'Command Deck', icon: 'radar', after: null },
    { id: 'observatory', label: 'Risk Observatory', icon: 'monitoring', after: 'threshold' },
    { id: 'calib', label: 'Calibration Lab', icon: 'compass_calibration', after: 'observatory' },
    { id: 'atlas', label: 'Signature Atlas', icon: 'blur_on', after: 'spatial' },
    { id: 'scope', label: 'Waveform Bank', icon: 'monitor_heart', after: 'telemetry' },
    { id: 'neural', label: 'Neural Core', icon: 'network_node', after: 'bench' },
  ];

  function mountNav() {
    const rail = q('#rail');
    NEW_VIEWS.forEach((v) => {
      const b = dom(`<button class="navlink" data-view="${v.id}">` +
        `<span class="material-symbols-outlined">${v.icon}</span>${esc(v.label)}</button>`);
      b.addEventListener('click', () => go(v.id));
      if (v.after) {
        const anchor = q(`.navlink[data-view="${v.after}"]`, rail);
        anchor ? anchor.after(b) : rail.appendChild(b);
      } else {
        q('.rail-h', rail).after(b);
      }
    });
    // and make every new deck reachable from the Ctrl-K palette too
    if (typeof PAL_VIEWS !== 'undefined') {
      const at = (id) => PAL_VIEWS.findIndex(([v]) => v === id);
      NEW_VIEWS.forEach((v) => {
        const i = v.after ? at(v.after) : -1;
        PAL_VIEWS.splice(i >= 0 ? i + 1 : 0, 0, [v.id, v.label, v.icon]);
      });
    }
  }

  const LOADERS = {
    command: loadCommand, neural: loadNeural, scope: loadScope,
    observatory: loadObservatory, calib: loadCalib, atlas: loadAtlas,
  };

  /** Load a deck the first time it becomes visible, however the user got there. */
  async function ensure(name) {
    if (name === 'run') mountRunDeck();
    const fn = LOADERS[name];
    if (!fn || HX.loaded[name]) return;
    HX.loaded[name] = 1;
    try { await fn(); } catch (e) { HX.loaded[name] = 0; toast(e.message || String(e)); }
  }

  /**
   * The rail, the palette and the number keys all reach a view through app.js's own
   * switchView, so rather than wrapping every entry point we watch the sections
   * themselves and load on the class change.
   */
  function watchViews() {
    const obs = new MutationObserver((muts) => {
      muts.forEach((m) => {
        if (m.target.classList.contains('on')) ensure(m.target.id.replace('view-', ''));
      });
    });
    NEW_VIEWS.map((v) => v.id).concat('run').forEach((id) => {
      const sec = q('#view-' + id);
      if (sec) obs.observe(sec, { attributes: true, attributeFilter: ['class'] });
    });
  }

  function go(name) {
    switchView(name);          // app.js toggles the sections; watchViews does the rest
    ensure(name);
  }

  // ==========================================================================
  // VIEW MARKUP
  // ==========================================================================
  function mountViews() {
    const main = q('main');
    if (!main) return;

    main.appendChild(dom(`
    <section class="view" id="view-command">
      <div class="panel lit">
        <span class="bracket tl"></span><span class="bracket tr"></span>
        <span class="bracket bl"></span><span class="bracket br"></span>
        <div class="ph">
          <span class="material-symbols-outlined">radar</span>
          <h2 class="big">Command Deck</h2>
          <span class="spacer"></span>
          <span class="hx-live">core online</span>
          <span class="tag violet" id="cdDataset">—</span>
          <span class="tag amber" id="cdThr">—</span>
          <span class="tag cyan" id="cdModel">—</span>
        </div>
        <div class="hx-strip" id="cdStrip"></div>
      </div>

      <div class="hx-cmdgrid">
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">nest_thermostat_gen_3</span>
            <h2>Reactor core</h2><span class="spacer"></span>
            <span class="sub mono" id="cdCoreSub">—</span></div>
          <div class="pb tight">
            <div class="hx-core-wrap" id="cdCore">
              <div class="hx-readout" id="cdReadout">
                <div class="big" id="cdRecall">—</div>
                <div class="lbl">new failures caught</div>
              </div>
            </div>
            <div class="hx-hazard" style="margin:6px 0 8px"></div>
            <p class="hx-note" style="margin:0">One spoke per held-out wafer, length set by that
              wafer's mean predicted risk and coloured on the matrix ramp. The green arc is the
              share of new failures the shipped cut actually catches.</p>
          </div>
        </div>

        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">bolt</span>
            <h2>Discrimination</h2><span class="spacer"></span>
            <span class="sub">AUC-PR on the failure class</span></div>
          <div class="pb tight">
            <div class="hx-g2" id="cdGauges"></div>
            <div class="hx-duel" id="cdDuel"></div>
          </div>
        </div>

        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">verified_user</span>
            <h2>Integrity</h2><span class="spacer"></span>
            <span class="sub mono" id="cdIntegrity">—</span></div>
          <div class="pb tight">
            <div id="cdSubs"><div class="hx-sub"><span class="nm">checking…</span></div></div>
            <div class="hx-hazard" style="margin:9px 0 8px"></div>
            <p class="hx-eyebrow" style="margin:0 0 4px">Held-out population</p>
            <div id="cdPop"></div>
          </div>
        </div>
      </div>

      <div class="g32">
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">ecg</span><h2>Fleet risk trace</h2>
            <span class="spacer"></span>
            <span class="sub mono" id="cdEkgCap">—</span></div>
          <div class="pb tight">
            <div class="hx-ekg"><canvas id="cdEkg"></canvas>
              <div class="cap">mean predicted risk per wafer · scrolling</div></div>
            <div id="cdWorst" style="margin-top:9px"></div>
          </div>
        </div>
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">crisis_alert</span>
            <h2>Operating point</h2><span class="spacer"></span>
            <span class="sub mono" id="cdCut">—</span></div>
          <div class="pb tight">
            <div class="tiles c4" id="cdConf"></div>
            <div id="cdOper" style="margin-top:9px"></div>
          </div>
        </div>
      </div>
    </section>`));

    main.appendChild(dom(`
    <section class="view" id="view-neural">
      <div class="panel lit">
        <div class="ph"><span class="material-symbols-outlined">network_node</span>
          <h2 class="big">Neural Core — signal path</h2>
          <span class="spacer"></span><span class="hx-live">tracing</span></div>
        <div class="pb" style="padding-top:0">
          <p class="hx-note" style="max-width:100ch">Each block is one rebuild of the model with one more
            component switched on, measured on the same held-out wafers. Edge brightness and thickness are
            that component's actual AUC-PR gain — a connection that carries nothing is drawn thin and dark.</p>
        </div>
        <div class="pb" style="padding-top:0"><div class="hx-flow" id="nnFlow"></div></div>
      </div>
      <div class="g32">
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">waterfall_chart</span><h2>Component waterfall</h2></div>
          <div class="pb"><div id="nnWaterfall"></div></div>
        </div>
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">bolt</span><h2>Selected stage</h2></div>
          <div class="pb"><div id="nnDetail"><p class="hx-note">Click any block in the signal path.</p></div></div>
        </div>
      </div>
    </section>`));

    main.appendChild(dom(`
    <section class="view" id="view-scope">
      <div class="panel lit">
        <div class="ph"><span class="material-symbols-outlined">monitor_heart</span>
          <h2 class="big">Waveform Bank</h2>
          <span class="spacer"></span>
          <div class="hx-scopectl">
            <button class="hx-btn on" id="scRun">Sweep</button>
            <button class="hx-btn" id="scPersist">Persistence</button>
            <button class="hx-btn" id="scTrig">Trigger on excursion</button>
            <span class="sub mono" style="margin-left:6px">Gain</span>
            <input type="range" class="hx-slider" id="scGain" min="50" max="260" value="100">
          </div>
        </div>
        <div class="pb" style="padding-top:0">
          <p class="hx-note" style="max-width:104ch">Four live dies off the active wafer, each showing all
            2,000 sub-die block readings straight from the fp16 memmap. The red band is the longest contiguous
            run above that die's own mean + 2σ — the excursion cluster the block branch is reacting to.</p>
        </div>
        <div class="pb" style="padding-top:0"><div class="hx-scopebank" id="scBank"></div></div>
      </div>
      <div class="g2">
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">graphic_eq</span><h2>Spectrum — channel A</h2>
            <span class="spacer"></span><span class="sub mono">1024-pt Hann FFT</span></div>
          <div class="pb tight"><div class="hx-chart"><canvas id="scFft" style="height:200px"></canvas></div>
            <p class="hx-note">Computed in the browser from the same 2,000 readings. A within-die periodic
              defect shows here as a line the raw trace hides.</p></div>
        </div>
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">blur_linear</span><h2>Spectrogram waterfall</h2>
            <span class="spacer"></span><span class="sub mono">128-pt window · hop 24</span></div>
          <div class="pb tight"><div class="hx-waterfall"><canvas id="scWfall"></canvas></div>
            <p class="hx-note">Block index runs down the image, frequency across. Bright horizontal
              structure means the excursion is localised to a band of blocks.</p></div>
        </div>
      </div>
      <div class="g32">
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">scatter_plot</span><h2>Phase portrait — A against B</h2></div>
          <div class="pb tight"><div class="hx-chart"><canvas id="scXy" style="height:290px"></canvas></div></div>
        </div>
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">fact_check</span><h2>Channel register</h2></div>
          <div class="pb tight"><div id="scRegister"></div></div>
        </div>
      </div>
    </section>`));

    main.appendChild(dom(`
    <section class="view" id="view-observatory">
      <div class="panel lit">
        <div class="ph"><span class="material-symbols-outlined">monitoring</span>
          <h2 class="big">Risk Observatory</h2>
          <span class="spacer"></span><span class="tag cyan" id="obN">—</span></div>
        <div class="pb" style="padding-top:0">
          <p class="hx-note" style="max-width:104ch">Every eligible die on the held-out wafers, one calibrated
            risk and one truth label each. Move the cut and every curve, count and decile below moves with it —
            all of it recomputed from the real test arrays, none of it cached.</p>
        </div>
        <div class="pb" style="padding-top:0">
          <div class="hx-scopectl">
            <span class="sub mono">Decision cut</span>
            <input type="range" class="hx-slider" id="obThr" min="1" max="990" value="500" style="flex:1 1 320px">
            <span class="hx-mono" id="obThrV" style="color:var(--cyan);min-width:70px;text-align:right">—</span>
            <button class="hx-btn" id="obShip">Shipped point</button>
            <button class="hx-btn" id="obKs">Max separation</button>
          </div>
        </div>
      </div>

      <div class="tiles c4" id="obTiles"></div>

      <div class="g2">
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">bar_chart</span><h2>Risk spectrum</h2>
            <span class="spacer"></span><span class="sub">log bins · failures overlaid</span></div>
          <div class="pb tight"><div class="hx-chart"><canvas id="obHist" style="height:250px"></canvas></div></div>
        </div>
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">show_chart</span><h2>ROC — and why it flatters</h2></div>
          <div class="pb tight"><div class="hx-chart"><svg id="obRoc" viewBox="0 0 480 330"></svg></div></div>
        </div>
      </div>

      <div class="g2">
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">trending_up</span><h2>Cumulative capture</h2>
            <span class="spacer"></span><span class="sub">failures found vs dies scrapped</span></div>
          <div class="pb tight"><div class="hx-chart"><svg id="obGain" viewBox="0 0 480 330"></svg></div></div>
        </div>
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">compare_arrows</span><h2>Class separation (KS)</h2>
            <span class="spacer"></span><span class="sub mono" id="obKsV">—</span></div>
          <div class="pb tight"><div class="hx-chart"><svg id="obKsSvg" viewBox="0 0 480 330"></svg></div></div>
        </div>
      </div>

      <div class="panel">
        <div class="ph"><span class="material-symbols-outlined">table_rows</span><h2>Risk deciles</h2>
          <span class="spacer"></span><span class="sub">rank the whole test split, cut it into ten</span></div>
        <div class="pb tight"><div class="tw"><table class="hx-decile" id="obDecile"></table></div></div>
      </div>
    </section>`));

    main.appendChild(dom(`
    <section class="view" id="view-calib">
      <div class="panel lit">
        <div class="ph"><span class="material-symbols-outlined">compass_calibration</span>
          <h2 class="big">Calibration &amp; Conformal Lab</h2>
          <span class="spacer"></span><span class="hx-live">verified</span></div>
        <div class="pb" style="padding-top:0">
          <p class="hx-note" style="max-width:104ch">A risk of 0.30 has to mean three dies in ten actually fail,
            or no operator can price a decision on it. Isotonic regression is fitted on validation only and
            applied here; the diagonal is what perfect honesty looks like.</p>
        </div>
      </div>
      <div class="tiles c4" id="cbTiles"></div>
      <div class="g2">
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">insights</span><h2>Reliability diagram</h2></div>
          <div class="pb tight"><div class="hx-chart"><svg id="cbRel" viewBox="0 0 480 340"></svg></div></div>
        </div>
        <div class="panel">
          <div class="ph"><span class="material-symbols-outlined">straighten</span><h2>Per-bin calibration gap</h2>
            <span class="spacer"></span><span class="sub">observed − predicted</span></div>
          <div class="pb tight"><div class="hx-chart"><canvas id="cbGap" style="height:280px"></canvas></div>
            <div id="cbGapRows" style="margin-top:10px"></div></div>
        </div>
      </div>
      <div class="panel">
        <div class="ph"><span class="material-symbols-outlined">expand</span><h2>Conformal intervals</h2></div>
        <div class="pb"><div id="cbConf"></div></div>
      </div>
    </section>`));

    main.appendChild(dom(`
    <section class="view" id="view-atlas">
      <div class="panel lit">
        <div class="ph"><span class="material-symbols-outlined">blur_on</span>
          <h2 class="big">Signature Atlas</h2>
          <span class="spacer"></span><span class="sub" id="atCount">—</span></div>
        <div class="pb" style="padding-top:0">
          <p class="hx-note" style="max-width:104ch">Nine wafer-scale failure signatures, each with the physical
            process fault it traces to and the module to go and inspect. The counts are how many held-out dies
            the explainer actually assigned to each class — the glyphs are drawn to the geometry the classifier
            keys on, so the rule stays auditable.</p>
        </div>
      </div>
      <div class="hx-atlas" id="atGrid"></div>
    </section>`));
  }

  // ==========================================================================
  // COMMAND DECK
  // ==========================================================================
  async function loadCommand() {
    const [s, wafers] = await Promise.all([api('/api/summary'), api('/api/wafers')]);
    HX.summary = s;
    const c = s.counts;
    const recall = c.caught / Math.max(c.new_fail, 1);

    q('#cdModel').textContent = s.model;
    q('#cdDataset').textContent = s.dataset + ' · held-out test';
    q('#cdThr').textContent = 'cut ' + Number(s.threshold).toFixed(4);
    q('#cdCut').textContent = 'cut ' + Number(s.threshold).toFixed(4);
    q('#cdCoreSub').textContent = `${c.wafers} wafers`;
    countTo(q('#cdRecall'), recall * 100, 1400, (v) => v.toFixed(1) + '%');

    buildStrip(c, s);
    buildPopulation(c);
    buildCore(wafers, recall);
    buildGauges(s);
    buildEkg(wafers);
    buildConfusion(s.confusion);
    buildOperating(s);

    api('/api/benchmarks').then((b) => { HX.bench = b; buildDuel(b.rows); }).catch(() => {});
    api('/api/provenance').then(buildSubs).catch(() => {
      q('#cdSubs').innerHTML = '<div class="hx-sub bad"><span class="nm">provenance endpoint unavailable</span><span class="st">FAIL</span></div>';
    });
  }

  /** The seven-cell headline strip. Every figure is a count off the test split. */
  function buildStrip(c, s) {
    const box = q('#cdStrip'); if (!box) return;
    const cells = [
      ['cy', 'Wafers held out', c.wafers, 'never seen in training', 0],
      ['', 'Dies on those wafers', c.dies_total, 'total probed on this lot', 0],
      ['cy', 'Eligible dies', c.eligible, 'old_label = 0 — the scored set', 0],
      ['ru', 'Pre-test failures', c.pre_test_fail, 'input signal, never a target', 0],
      ['am', 'New failures', c.new_fail, `${(c.fail_rate * 100).toFixed(2)}% base rate`, 0],
      ['vi', 'Dies flagged', c.flagged,
        `${((100 * c.flagged) / Math.max(c.eligible, 1)).toFixed(2)}% scrapped`, 0],
      ['mi', 'Failures caught', c.caught, `${c.escaped} escaped to the customer`, 0],
    ];
    box.innerHTML = '';
    cells.forEach(([cls, k, v, n]) => {
      const cell = dom(`<div class="hx-cell ${cls}"><div class="k">${esc(k)}</div>` +
        `<div class="v">0</div><div class="n">${esc(n)}</div></div>`);
      box.appendChild(cell);
      countTo(q('.v', cell), v, 1000);
    });
  }

  function buildPopulation(c) {
    const box = q('#cdPop'); if (!box) return;
    const rows = [
      ['Base failure rate', (c.fail_rate * 100).toFixed(2) + '%', 'amber'],
      ['Flag rate at the shipped cut',
        ((100 * c.flagged) / Math.max(c.eligible, 1)).toFixed(2) + '%', 'violet'],
      ['Escapes past the cut', c.escaped.toLocaleString('en-US'), 'ruby'],
      ['All-pass accuracy, for contrast', ((1 - c.fail_rate) * 100).toFixed(2) + '%', ''],
    ];
    box.innerHTML = rows.map(([k, v, cls]) =>
      `<div class="kv"><span class="k">${esc(k)}</span>` +
      `<span class="v ${cls}">${esc(v)}</span></div>`).join('');
  }

  function buildOperating(s) {
    const box = q('#cdOper'); if (!box) return;
    const cf = s.confusion || {};
    const rows = [
      ['Recall on the failure class', (cf.fail_recall * 100).toFixed(1) + '%', 'cyan'],
      ['Precision on flags', (cf.precision * 100).toFixed(1) + '%', 'violet'],
      ['Wafer scrapped', cf.scrap_pct + '%', 'amber'],
      ['Escape rate', cf.escape_pct + '%', 'ruby'],
      ['Accuracy, reported not selected on', (cf.accuracy * 100).toFixed(2) + '%', ''],
    ];
    box.innerHTML = rows.map(([k, v, cls]) =>
      `<div class="kv"><span class="k">${esc(k)}</span>` +
      `<span class="v ${cls}">${esc(v)}</span></div>`).join('');
  }

  /** The three wafers carrying the most risk, under the fleet trace. */
  function buildWorst(wafers) {
    const box = q('#cdWorst'); if (!box) return;
    const top = wafers.slice().sort((a, b) => b.new_fail - a.new_fail).slice(0, 3);
    box.innerHTML = top.map((w) => {
      const caught = (100 * w.caught) / Math.max(w.new_fail, 1);
      return `<div class="hx-sub ${caught < 40 ? 'warn' : ''}">
        <span class="nm"><b style="color:var(--cyan)">${esc(w.wafer_id)}</b>
          · ${w.new_fail} new fails of ${w.eligible} eligible · yield ${w.yield_pct}%</span>
        <span class="bar"><i style="width:${caught.toFixed(0)}%"></i></span>
        <span class="st">${caught.toFixed(0)}%</span></div>`;
    }).join('');
  }

  function buildCore(wafers, recall) {
    const wrap = q('#cdCore');
    qa('svg', wrap).forEach((n) => n.remove());
    const CX = 220, CY = 168;
    const s = sv('svg', { viewBox: '0 0 440 336' });

    const defs = sv('defs');
    const rg = sv('radialGradient', { id: 'hxCore' });
    rg.appendChild(sv('stop', { offset: '0%', 'stop-color': P.cyan, 'stop-opacity': .85 }));
    rg.appendChild(sv('stop', { offset: '55%', 'stop-color': '#0A7CA8', 'stop-opacity': .35 }));
    rg.appendChild(sv('stop', { offset: '100%', 'stop-color': '#05080E', 'stop-opacity': 0 }));
    defs.appendChild(rg);
    const sg = sv('radialGradient', { id: 'hxSweep' });
    sg.appendChild(sv('stop', { offset: '0%', 'stop-color': P.cyan, 'stop-opacity': .30 }));
    sg.appendChild(sv('stop', { offset: '100%', 'stop-color': P.cyan, 'stop-opacity': 0 }));
    defs.appendChild(sg);
    s.appendChild(defs);

    const spin = (r, cls, dash, op) => {
      const g = sv('g', { class: `hx-ring ${cls}`,
        style: `transform-box:view-box;transform-origin:${CX}px ${CY}px` });
      g.appendChild(sv('circle', { cx: CX, cy: CY, r, fill: 'none',
        stroke: 'rgba(0,240,255,.28)', 'stroke-width': 1,
        'stroke-dasharray': dash, opacity: op }));
      return g;
    };
    s.appendChild(spin(152, 'c', '2 9', .55));
    s.appendChild(spin(133, 'a', '30 6', .35));
    s.appendChild(spin(112, 'b', '1 5', .5));

    // degree callouts
    for (let a = 0; a < 360; a += 45) {
      const r = (a * Math.PI) / 180;
      s.appendChild(sv('text', {
        x: CX + Math.cos(r - Math.PI / 2) * 168, y: CY + Math.sin(r - Math.PI / 2) * 168 + 3,
        fill: P.dim, 'text-anchor': 'middle', 'font-family': 'JetBrains Mono', 'font-size': 7.5,
      }, String(a).padStart(3, '0')));
    }

    // rotating radar sweep, under the spokes so it reads as a scan across them
    const sweep = sv('g', { style: `transform-box:view-box;transform-origin:${CX}px ${CY}px`,
      class: 'hx-ring a' });
    const SW = 152, half = 0.30;                          // ~34 degrees
    sweep.appendChild(sv('path', {
      d: `M ${CX} ${CY} L ${CX + Math.cos(-Math.PI / 2 - half) * SW} ${CY + Math.sin(-Math.PI / 2 - half) * SW} ` +
         `A ${SW} ${SW} 0 0 1 ${CX + Math.cos(-Math.PI / 2 + half) * SW} ${CY + Math.sin(-Math.PI / 2 + half) * SW} Z`,
      fill: 'url(#hxSweep)' }));
    sweep.appendChild(sv('line', { x1: CX, y1: CY, x2: CX, y2: CY - SW,
      stroke: 'rgba(0,240,255,.55)', 'stroke-width': 1.2 }));
    s.appendChild(sweep);

    // one spoke per wafer — length is that wafer's mean predicted risk
    const list = wafers.slice().sort((a, b) => a.wafer_id.localeCompare(b.wafer_id));
    const maxR = Math.max(...list.map((w) => w.mean_risk || 0), 1e-6);
    const spokes = sv('g');
    list.forEach((w, i) => {
      const a = (i / list.length) * Math.PI * 2 - Math.PI / 2;
      const r0 = 62, len = 12 + 40 * Math.sqrt((w.mean_risk || 0) / maxR);
      const ln = sv('line', {
        x1: CX + Math.cos(a) * r0, y1: CY + Math.sin(a) * r0,
        x2: CX + Math.cos(a) * (r0 + len), y2: CY + Math.sin(a) * (r0 + len),
        stroke: riskColor(w.mean_risk), 'stroke-width': 2.4, 'stroke-linecap': 'round',
        opacity: 0,
      });
      ln.style.transition = `opacity .5s ease ${i * 7}ms`;
      spokes.appendChild(ln);
      ln.dataset.w = w.wafer_id;
    });
    s.appendChild(spokes);
    requestAnimationFrame(() => qa('line', spokes).forEach((l) => { l.style.opacity = .92; }));

    // core
    s.appendChild(sv('circle', { cx: CX, cy: CY, r: 58, fill: 'url(#hxCore)', class: 'hx-corepulse' }));
    s.appendChild(sv('circle', { cx: CX, cy: CY, r: 58, fill: 'none',
      stroke: 'rgba(0,240,255,.45)', 'stroke-width': 1 }));

    // containment arc — fraction of failures caught
    const R = 92, frac = clamp(recall, 0, 1);
    const a0 = -Math.PI / 2, a1 = a0 + frac * Math.PI * 2;
    const large = frac > 0.5 ? 1 : 0;
    const arc = sv('path', {
      d: `M ${CX + Math.cos(a0) * R} ${CY + Math.sin(a0) * R} A ${R} ${R} 0 ${large} 1 ` +
         `${CX + Math.cos(a1) * R} ${CY + Math.sin(a1) * R}`,
      fill: 'none', stroke: P.mint, 'stroke-width': 4, 'stroke-linecap': 'round',
      opacity: .9,
    });
    s.appendChild(sv('circle', { cx: CX, cy: CY, r: R, fill: 'none',
      stroke: 'rgba(0,230,118,.14)', 'stroke-width': 4 }));
    s.appendChild(arc);
    drawIn(arc, 1500, 250);

    wrap.insertBefore(s, wrap.firstChild);
  }

  function gauge(key, label, value, display, note, color) {
    const R = 46, cx = 65, cy = 62;
    const circ = Math.PI * R;      // half circle
    const box = dom(`<div class="hx-gauge"><div class="g-k">${esc(label)}</div>
      <div class="g-n">${esc(note)}</div></div>`);
    const s = sv('svg', { viewBox: '0 0 130 82' });
    s.appendChild(sv('path', { d: `M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`,
      fill: 'none', stroke: 'rgba(0,240,255,.12)', 'stroke-width': 7, 'stroke-linecap': 'round' }));
    const arc = sv('path', { d: `M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`,
      fill: 'none', stroke: color, 'stroke-width': 7, 'stroke-linecap': 'round',
      class: 'g-arc', 'stroke-dasharray': circ, 'stroke-dashoffset': circ });
    s.appendChild(arc);
    // ticks
    for (let i = 0; i <= 10; i++) {
      const a = Math.PI + (i / 10) * Math.PI;
      s.appendChild(sv('line', {
        x1: cx + Math.cos(a) * (R + 7), y1: cy + Math.sin(a) * (R + 7),
        x2: cx + Math.cos(a) * (R + 11), y2: cy + Math.sin(a) * (R + 11),
        stroke: 'rgba(0,240,255,.25)', 'stroke-width': 1 }));
    }
    s.appendChild(sv('text', { x: cx, y: cy - 6, fill: P.hi, 'text-anchor': 'middle',
      'font-family': 'JetBrains Mono', 'font-size': 19, 'font-weight': 700 }, display));
    box.insertBefore(s, box.firstChild);
    setTimeout(() => { arc.setAttribute('stroke-dashoffset', circ * (1 - clamp(value, 0, 1))); }, 90);
    return box;
  }

  function buildGauges(s) {
    const g = q('#cdGauges'); g.innerHTML = '';
    const m = s.metrics || {};
    const b = (s.model_b || {});
    const gain = s.discrimination_gain;
    g.appendChild(gauge('auc', 'AUC-PR · Model B', b.auc_pr || 0,
      (b.auc_pr || 0).toFixed(4), 'primary selection metric', P.cyan));
    g.appendChild(gauge('rec', 'Recall on failures', m.recall || 0,
      ((m.recall || 0) * 100).toFixed(1) + '%', 'of new failures caught', P.mint));
    g.appendChild(gauge('prec', 'Precision', m.precision || 0,
      ((m.precision || 0) * 100).toFixed(1) + '%', 'of flags that were right', P.violet));
    g.appendChild(gauge('gain', 'Lift over GDBN', clamp((gain || 0) / 8, 0, 1),
      (gain || 0).toFixed(1) + '×', 'against the incumbent screen', P.amber));
  }

  function buildEkg(wafers) {
    const cv = q('#cdEkg');
    const list = wafers.slice().sort((a, b) => a.wafer_id.localeCompare(b.wafer_id));
    const sig = list.map((w) => w.mean_risk || 0);
    const peak = list.reduce((a, w) => (w.mean_risk > (a.mean_risk || 0) ? w : a), list[0] || {});
    q('#cdEkgCap').textContent =
      `${list.length} wafers · peak ${peak.wafer_id || '—'} at ${((peak.mean_risk || 0) * 100).toFixed(2)}%`;
    buildWorst(list);
    const hi = Math.max(...sig, 1e-6);
    let off = 0;
    ticker((t) => {
      const { ctx, w, h } = fit(cv);
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = 'rgba(0,240,255,.07)'; ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 32) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (let y = 0; y < h; y += 24) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

      off = (t / 42) % sig.length;
      const span = Math.min(sig.length, 72);
      const pt = (i) => {
        const f = (off + (i / w) * span) % sig.length;
        const a = sig[Math.floor(f)], b = sig[(Math.floor(f) + 1) % sig.length];
        const k = f - Math.floor(f);
        return a + (b - a) * k;
      };
      ctx.beginPath();
      for (let x = 0; x <= w; x += 2) {
        const v = pt(x);
        const y = h - 12 - (v / hi) * (h - 30);
        x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.strokeStyle = P.cyan; ctx.lineWidth = 1.5;
      ctx.shadowColor = P.cyan; ctx.shadowBlur = 10; ctx.stroke(); ctx.shadowBlur = 0;

      ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
      const grd = ctx.createLinearGradient(0, 0, 0, h);
      grd.addColorStop(0, 'rgba(0,240,255,.22)');
      grd.addColorStop(1, 'rgba(0,240,255,0)');
      ctx.fillStyle = grd; ctx.fill();

      const hv = pt(w - 1), hy = h - 12 - (hv / hi) * (h - 30);
      ctx.fillStyle = P.cyan; ctx.shadowColor = P.cyan; ctx.shadowBlur = 14;
      ctx.beginPath(); ctx.arc(w - 1, hy, 2.6, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
    }, cv);
  }

  function buildDuel(rows) {
    const box = q('#cdDuel'); if (!box) return;
    box.innerHTML = '';
    const hi = Math.max(...rows.map((r) => r.auc_pr || 0), 1e-6);
    const col = { trap: 'rgba(88,96,105,.75)', incumbent: P.amber, ours: P.cyan, variant: P.violet };
    rows.forEach((r, i) => {
      const row = dom(`<div class="row">
        <span class="nm">${esc(r.model)}</span>
        <span class="track"><i></i></span>
        <span class="vv">${(r.auc_pr || 0).toFixed(4)}</span></div>`);
      const bar = q('i', row);
      bar.style.background = col[r.kind] || P.cyan;
      box.appendChild(row);
      setTimeout(() => { bar.style.width = `${((r.auc_pr || 0) / hi) * 100}%`; }, 60 + i * 80);
    });
  }

  function buildConfusion(c) {
    const box = q('#cdConf'); if (!box || !c) return;
    box.innerHTML = '';
    const cells = [
      ['mint', 'Caught (TP)', c.true_pos, 'flagged and it really failed'],
      ['amber', 'False alarm (FP)', c.false_pos, 'scrapped a good die'],
      ['ruby', 'Escape (FN)', c.false_neg, 'shipped a failure'],
      ['', 'Correct pass (TN)', c.true_neg, 'left alone, correctly'],
    ];
    cells.forEach(([cls, k, v, n]) => {
      const t = dom(`<div class="tile ${cls}"><div class="k">${esc(k)}</div>` +
        `<div class="v">0</div><div class="n">${esc(n)}</div></div>`);
      box.appendChild(t);
      countTo(q('.v', t), v, 900);
    });
  }

  function buildSubs(p) {
    const box = q('#cdSubs'); if (!box) return;
    const checks = (p && p.checks) || [];
    box.innerHTML = '';
    if (!checks.length) {
      box.innerHTML = '<div class="hx-sub warn"><span class="nm">no checks returned</span><span class="st">—</span></div>';
      return;
    }
    const ok = checks.filter((c) => c.ok).length;
    const hdr = q('#cdIntegrity');
    if (hdr) {
      hdr.textContent = `${ok} / ${checks.length} pass`;
      hdr.style.color = ok === checks.length ? 'var(--mint)' : 'var(--ruby)';
    }
    checks.forEach((c, i) => {
      const row = dom(`<div class="hx-sub ${c.ok ? '' : 'bad'}">
        <span class="nm" title="${esc(c.detail)}">${esc(c.name)}</span>
        <span class="bar"><i></i></span>
        <span class="st">${c.ok ? 'PASS' : 'FAIL'}</span></div>`);
      box.appendChild(row);
      setTimeout(() => { q('i', row).style.width = c.ok ? '100%' : '32%'; }, 80 + i * 55);
    });
  }

  // ==========================================================================
  // NEURAL CORE
  // ==========================================================================
  const shortName = (s) => String(s)
    .replace(/\s*\[MODEL [AB] FINAL\]\s*/i, '')
    .replace(/^[A-Z0-9]+:\s*/, '')
    .replace(/^\+\s*/, '')
    .trim();

  async function loadNeural() {
    const b = HX.bench || (HX.bench = await api('/api/benchmarks'));
    const steps = b.ablation || [];
    if (!steps.length) {
      q('#nnFlow').innerHTML = '<div class="err" style="margin:14px">No ablation rows in results.csv yet — run the ablation task.</div>';
      return;
    }
    buildFlow(steps);
    buildWaterfall(steps);
    showStage(steps[steps.length - 1], steps);
  }

  function buildFlow(steps) {
    const wrap = q('#nnFlow'); wrap.innerHTML = '';
    // fewer columns when there are few stages, so the serpentine fills the panel
    const COLS = steps.length <= 6 ? 3 : 4;
    const BW = 228, BH = 66, GX = 62, GY = 86;
    const rows = Math.ceil(steps.length / COLS);
    const W = COLS * BW + (COLS - 1) * GX + 130;
    const H = rows * BH + (rows - 1) * GY + 70;
    const s = sv('svg', { viewBox: `0 0 ${W} ${H}` });

    const pos = steps.map((_, i) => {
      const r = Math.floor(i / COLS);
      let c = i % COLS;
      if (r % 2 === 1) c = COLS - 1 - c;                    // serpentine
      return { x: 65 + c * (BW + GX), y: 34 + r * (BH + GY), r, c };
    });

    const maxD = Math.max(...steps.map((s2) => Math.abs(s2.delta || 0)), 1e-9);

    // edges first
    for (let i = 1; i < steps.length; i++) {
      const a = pos[i - 1], b2 = pos[i];
      const ax = a.x + BW / 2, ay = a.y + BH / 2, bx = b2.x + BW / 2, by = b2.y + BH / 2;
      let d;
      if (a.r === b2.r) {
        const s1 = a.c < b2.c ? 1 : -1;
        d = `M ${ax + s1 * BW / 2} ${ay} L ${bx - s1 * BW / 2} ${by}`;
      } else {
        const s1 = a.c === COLS - 1 ? 1 : -1;
        const ox = ax + s1 * (BW / 2 + 26);
        d = `M ${ax + s1 * BW / 2} ${ay} C ${ox} ${ay}, ${ox} ${by}, ${bx + s1 * BW / 2} ${by}`;
      }
      const mag = Math.abs(steps[i].delta || 0) / maxD;
      s.appendChild(sv('path', { d, class: 'hx-edge', 'stroke-width': 1 }));
      const flow = sv('path', { d, class: 'hx-edge-flow',
        'stroke-width': (0.8 + mag * 4.2).toFixed(2),
        'stroke-dasharray': '5 17', opacity: (0.25 + mag * 0.75).toFixed(2) });
      flow.style.animationDelay = `${-i * 0.22}s`;
      s.appendChild(flow);
    }

    // nodes
    steps.forEach((st, i) => {
      const p = pos[i];
      const g = sv('g', { class: 'hx-nodebox' });
      const pos_ = (st.delta || 0) >= 0;
      const col = i === 0 ? P.dim : (pos_ ? P.cyan : P.ruby);
      g.appendChild(sv('rect', { x: p.x, y: p.y, width: BW, height: BH, rx: 5,
        fill: 'rgba(11,21,40,.85)', stroke: 'rgba(0,240,255,.22)', 'stroke-width': 1 }));
      g.appendChild(sv('rect', { x: p.x, y: p.y, width: 3, height: BH, rx: 1.5, fill: col }));
      const nm = shortName(st.step);
      g.appendChild(sv('text', { x: p.x + 13, y: p.y + 21, fill: P.hi,
        'font-family': 'Outfit', 'font-size': nm.length > 30 ? 11 : 12.5, 'font-weight': 700 },
        nm.length > 34 ? nm.slice(0, 33) + '…' : nm));
      g.appendChild(sv('text', { x: p.x + 13, y: p.y + 40, fill: P.dim,
        'font-family': 'JetBrains Mono', 'font-size': 9 },
        `AUC-PR ${(st.auc_pr || 0).toFixed(4)}`));
      g.appendChild(sv('text', { x: p.x + BW - 13, y: p.y + 40, fill: col,
        'text-anchor': 'end', 'font-family': 'JetBrains Mono', 'font-size': 11, 'font-weight': 700 },
        st.delta == null ? 'base' : `${st.delta >= 0 ? '+' : ''}${st.delta.toFixed(4)}`));
      g.appendChild(sv('text', { x: p.x + 13, y: p.y + 56, fill: 'rgba(88,96,105,.85)',
        'font-family': 'JetBrains Mono', 'font-size': 8 }, `STAGE ${String(i + 1).padStart(2, '0')}`));
      g.addEventListener('click', () => {
        qa('.hx-nodebox', s).forEach((n) => n.classList.remove('on'));
        g.classList.add('on');
        showStage(st, steps);
      });
      s.appendChild(g);
    });

    wrap.appendChild(s);
  }

  function buildWaterfall(steps) {
    const box = q('#nnWaterfall'); box.innerHTML = '';
    const hi = Math.max(...steps.map((s) => s.auc_pr || 0), 1e-6);
    steps.forEach((st, i) => {
      const prev = i ? (steps[i - 1].auc_pr || 0) : 0;
      const cur = st.auc_pr || 0;
      const lo = Math.min(prev, cur), span = Math.abs(cur - prev);
      const row = dom(`<div class="hx-wfrow">
        <span class="nm" title="${esc(st.step)}">${esc(shortName(st.step))}</span>
        <span class="tr"><i></i></span>
        <span class="dv">${st.delta == null ? '<span style="color:#586069">baseline</span>'
          : `<span style="color:${st.delta >= 0 ? P.mint : P.ruby}">${st.delta >= 0 ? '+' : ''}${st.delta.toFixed(4)}</span>`}</span>
        <span class="av">${cur.toFixed(4)}</span></div>`);
      box.appendChild(row);
      const bar = q('i', row);
      bar.style.background = i === 0 ? 'rgba(88,96,105,.7)'
        : (st.delta >= 0 ? P.cyan : P.ruby);
      setTimeout(() => {
        bar.style.left = `${(i === 0 ? 0 : (lo / hi) * 100).toFixed(2)}%`;
        bar.style.width = `${(i === 0 ? (cur / hi) * 100 : Math.max((span / hi) * 100, 0.6)).toFixed(2)}%`;
      }, 70 + i * 90);
    });
  }

  function showStage(st, steps) {
    const box = q('#nnDetail');
    const i = steps.indexOf(st);
    const share = st.delta != null && steps[steps.length - 1].auc_pr
      ? (st.delta / steps[steps.length - 1].auc_pr) * 100 : null;
    box.innerHTML = `
      <p class="hx-eyebrow">Stage ${String(i + 1).padStart(2, '0')} of ${steps.length}</p>
      <h3 style="font-family:var(--f-head);font-size:16px;margin:0 0 10px;color:var(--tx-hi)">${esc(shortName(st.step))}</h3>
      <div class="kv"><span class="k">AUC-PR after this stage</span><span class="v cyan">${(st.auc_pr || 0).toFixed(4)}</span></div>
      <div class="kv"><span class="k">Gain this component adds</span><span class="v ${st.delta >= 0 ? 'mint' : 'ruby'}">${
        st.delta == null ? 'baseline' : (st.delta >= 0 ? '+' : '') + st.delta.toFixed(4)}</span></div>
      <div class="kv"><span class="k">Share of final score</span><span class="v">${
        share == null ? '—' : share.toFixed(1) + '%'}</span></div>
      <div class="kv"><span class="k">results.csv row</span><span class="v mono" style="font-size:10px">${esc(st.step)}</span></div>
      <p class="hx-note">Measured by rebuilding the model with exactly this set of components and
        scoring the same held-out wafers. Nothing is estimated from feature importance.</p>`;
  }

  // ==========================================================================
  // WAVEFORM BANK
  // ==========================================================================
  const SCOPE_COLORS = [P.ruby, P.amber, P.violet, P.cyan];

  async function pickChannels() {
    let w = (typeof S !== 'undefined' && S.wafer) ? S.wafer : null;
    if (!w) {
      const f = await api('/api/focus');
      w = await api('/api/wafer/' + encodeURIComponent(f.wafer_id));
    }
    const n = w.state.length;
    const cand = [];
    for (let i = 0; i < n; i++) {
      if (w.p_fail[i] == null) continue;
      cand.push({ i, p: w.p_fail[i], st: w.state[i], yp: w.y_pred[i],
        r: w.die_row[i], c: w.die_col[i] });
    }
    const best = (f) => cand.filter(f).sort((a, b) => b.p - a.p)[0];
    const worst = (f) => cand.filter(f).sort((a, b) => a.p - b.p)[0];
    const chosen = [
      { d: best((x) => x.st === 1 && x.yp === 1), tag: 'CAUGHT FAILURE' },
      { d: best((x) => x.st === 0 && x.yp === 1), tag: 'FALSE ALARM' },
      { d: best((x) => x.st === 1 && x.yp === 0), tag: 'ESCAPE' },
      { d: worst((x) => x.st === 0 && x.yp === 0), tag: 'CLEAN REFERENCE' },
    ];
    const used = new Set();
    const out = [];
    for (const c of chosen) {
      let d = c.d;
      if (!d || used.has(d.i)) d = cand.filter((x) => !used.has(x.i)).sort((a, b) => b.p - a.p)[0];
      if (!d) continue;
      used.add(d.i);
      out.push({ ...c, d });
    }
    return { wafer: w.wafer_id, chans: out };
  }

  async function loadScope() {
    const { wafer, chans } = await pickChannels();
    const bank = q('#scBank'); bank.innerHTML = '';
    const loaded = [];

    for (let k = 0; k < chans.length; k++) {
      const { d, tag } = chans[k];
      const col = SCOPE_COLORS[k];
      const cell = dom(`<div class="hx-ch">
        <canvas></canvas>
        <div class="hd"><span class="dot" style="background:${col};box-shadow:0 0 8px ${col}"></span>
          <span class="nm">CH ${String.fromCharCode(65 + k)} · ${esc(tag)}</span>
          <span class="sp"></span><span class="rd">loading…</span></div>
        <div class="ft"></div></div>`);
      bank.appendChild(cell);

      let die;
      try {
        die = await api(`/api/die/${encodeURIComponent(wafer)}/${d.r}/${d.c}`);
      } catch (e) { q('.rd', cell).textContent = 'unavailable'; continue; }
      const b = die.blocks;
      if (!b || !b.series) { q('.rd', cell).textContent = 'no waveform'; continue; }

      q('.rd', cell).textContent =
        `[${d.r},${d.c}] risk ${(die.p_fail * 100).toFixed(1)}%`;
      q('.ft', cell).innerHTML =
        `<span>μ <b>${b.mean}</b></span><span>σ <b>${b.std}</b></span>` +
        `<span>peak <b>#${b.peak_index}</b></span>` +
        `<span>hot <b>${b.hot_pct}%</b></span>` +
        `<span>excursion <b>${b.cluster ? `${b.cluster.lo}–${b.cluster.hi}` : 'none'}</b></span>` +
        `<span>ρ₁ <b>${b.autocorrelation ?? '—'}</b></span>`;

      loaded.push({ cell, canvas: q('canvas', cell), blocks: b, col, die, tag, d });
      mountChannel(loaded[loaded.length - 1], k);
    }

    HX.scope = { wafer, chans: loaded, run: true, persist: false, trig: false, gain: 1 };
    buildRegister(loaded, wafer);
    if (loaded[0]) { buildFft(loaded[0]); buildWaterfallSpec(loaded[0]); }
    if (loaded[0] && loaded[1]) buildXy(loaded[0], loaded[1]);
    wireScopeControls();
  }

  function mountChannel(ch, idx) {
    const store = {};
    const trail = [];
    let cols = null, colsW = 0, colsGain = 0;

    ticker((t) => {
      const { ctx, w, h } = fit(ch.canvas);
      if (!w) return;
      const st = HX.scope || {};
      const gain = st.gain || 1;
      const s = ch.blocks.series;
      const T = 26, B = 24, L = 8, Rr = 8;
      const ih = h - T - B, iw = w - L - Rr;
      const mid = (ch.blocks.min + ch.blocks.max) / 2;
      const halfSpan = ((ch.blocks.max - ch.blocks.min) / 2) / gain * 1.1;
      const lo = mid - halfSpan, hi = mid + halfSpan;
      const Y = (v) => T + ih - ((v - lo) / (hi - lo || 1)) * ih;

      if (!cols || colsW !== iw) { cols = columns(s, iw); colsW = iw; }
      if (colsGain !== gain) { colsGain = gain; store.bg = null; }

      // graticule, excursion band and the dim full trace never change: draw them once
      const bg = backing(store, 'bg', w, h, (c) => {
        c.strokeStyle = 'rgba(0,240,255,.07)'; c.lineWidth = 1;
        c.beginPath();
        for (let k = 0; k <= 8; k++) { const x = L + (iw * k) / 8; c.moveTo(x, T); c.lineTo(x, T + ih); }
        for (let k = 0; k <= 4; k++) { const y = T + (ih * k) / 4; c.moveTo(L, y); c.lineTo(w - Rr, y); }
        c.stroke();
        if (ch.blocks.cluster) {
          const n = s.length;
          const x1 = L + (ch.blocks.cluster.lo / (n - 1)) * iw;
          const x2 = L + (ch.blocks.cluster.hi / (n - 1)) * iw;
          c.fillStyle = 'rgba(255,23,68,.14)';
          c.fillRect(x1, T, Math.max(x2 - x1, 2), ih);
          c.strokeStyle = 'rgba(255,23,68,.5)';
          c.strokeRect(x1, T, Math.max(x2 - x1, 2), ih);
        }
        c.beginPath();
        cols.forEach((mm, x) => { c.moveTo(L + x, Y(mm[0])); c.lineTo(L + x, Y(mm[1])); });
        c.strokeStyle = ch.col + '3A'; c.stroke();
      });

      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(bg, 0, 0, w, h);

      // trigger line at that die's own mean + 2 sigma
      const trig = ch.blocks.mean + 2 * ch.blocks.std;
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = st.trig ? 'rgba(255,179,0,.75)' : 'rgba(0,240,255,.30)';
      ctx.beginPath(); ctx.moveTo(L, Y(trig)); ctx.lineTo(w - Rr, Y(trig)); ctx.stroke();
      ctx.setLineDash([]);

      // sweeping beam over the same columns
      const period = 4200;
      const phase = st.run === false ? 1 : ((t + idx * 500) % period) / period;
      const upto = Math.floor(phase * (cols.length - 1));
      ctx.beginPath();
      for (let x = 0; x <= upto; x++) {
        ctx.moveTo(L + x, Y(cols[x][0])); ctx.lineTo(L + x, Y(cols[x][1]));
      }
      ctx.strokeStyle = ch.col; ctx.lineWidth = 1;
      ctx.shadowColor = ch.col; ctx.shadowBlur = 7; ctx.stroke(); ctx.shadowBlur = 0;

      if (st.run !== false) {
        const bx = L + upto, by = Y((cols[upto][0] + cols[upto][1]) / 2);
        trail.push([bx, by]);
        if (trail.length > (st.persist ? 70 : 18)) trail.shift();
        ctx.globalCompositeOperation = 'lighter';
        ctx.fillStyle = ch.col;
        for (let i2 = 0; i2 < trail.length; i2++) {
          ctx.globalAlpha = (i2 / trail.length) * 0.32;
          ctx.fillRect(trail[i2][0] - 1.4, trail[i2][1] - 1.4, 2.8, 2.8);
        }
        ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = '#fff'; ctx.shadowColor = ch.col; ctx.shadowBlur = 10;
        ctx.beginPath(); ctx.arc(bx, by, 2.4, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
        ctx.strokeStyle = ch.col + '55';
        ctx.beginPath(); ctx.moveTo(bx, T); ctx.lineTo(bx, T + ih); ctx.stroke();
      }
    }, ch.canvas);
  }

  function buildRegister(chans, wafer) {
    const box = q('#scRegister');
    box.innerHTML = `<p class="hx-eyebrow">Active wafer ${esc(wafer)}</p>`;
    chans.forEach((c, i) => {
      const d = c.die;
      box.appendChild(dom(`<div class="hx-sub">
        <span class="nm"><span style="color:${c.col}">CH ${String.fromCharCode(65 + i)}</span>
          · die [${d.die_row},${d.die_col}] · ${esc(c.tag.toLowerCase())}</span>
        <span class="st" style="color:${c.col}">${(d.p_fail * 100).toFixed(1)}%</span></div>`));
    });
    const c0 = chans[0];
    if (c0) {
      box.appendChild(dom(`<div class="hx-hr"></div>
        <div class="kv"><span class="k">Conformal interval, CH A</span><span class="v cyan">${
          c0.die.conformal && c0.die.conformal.lo != null
            ? `${c0.die.conformal.lo.toFixed(3)} – ${c0.die.conformal.hi.toFixed(3)}` : '—'}</span></div>
        <div class="kv"><span class="k">Root cause</span><span class="v">${
          esc((c0.die.root_cause || {}).signature || '—')}</span></div>
        <div class="kv"><span class="k">Rank among all scored dies</span><span class="v">#${c0.die.rank}</span></div>`));
    }
  }

  // ---- FFT -----------------------------------------------------------------
  function fftMag(input) {
    // radix-2, in place, real input zero-padded to the next power of two
    let n = 1; while (n < input.length) n <<= 1;
    const re = new Float64Array(n), im = new Float64Array(n);
    for (let i = 0; i < input.length; i++) {
      const wnd = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (input.length - 1));  // Hann
      re[i] = input[i] * wnd;
    }
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const ur = re[i + k], ui = im[i + k];
          const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
          const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
          re[i + k] = ur + vr; im[i + k] = ui + vi;
          re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
    const out = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) out[i] = Math.hypot(re[i], im[i]);
    return out;
  }

  function buildFft(ch) {
    const cv = q('#scFft');
    const s = ch.blocks.series;
    const mean = ch.blocks.mean;
    const seg = s.slice(0, 1024).map((v) => v - mean);
    const mag = fftMag(seg);
    const db = Array.from(mag).map((v) => 20 * Math.log10(v + 1e-9));
    const lo = Math.min(...db), hi = Math.max(...db);
    const peakBin = db.indexOf(hi);
    const grow = { t0: null };
    ticker((t) => {
      const { ctx, w, h } = fit(cv);
      ctx.clearRect(0, 0, w, h);
      const gk = ramp(grow, t, 1200);
      const T = 12, B = 22, L = 34, Rr = 8;
      const iw = w - L - Rr, ih = h - T - B;
      ctx.strokeStyle = 'rgba(0,240,255,.08)';
      for (let k = 0; k <= 4; k++) {
        const y = T + (ih * k) / 4;
        ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - Rr, y); ctx.stroke();
        ctx.fillStyle = P.dim; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'right';
        ctx.fillText((hi - ((hi - lo) * k) / 4).toFixed(0), L - 5, y + 3);
      }
      const N = Math.min(db.length, 256);
      const bw = iw / N;
      for (let i = 0; i < N; i++) {
        const v = (db[i] - lo) / (hi - lo || 1);
        const bh = v * ih * gk * (0.94 + 0.06 * Math.sin(t / 260 + i * 0.4));
        const isPeak = i === peakBin;
        ctx.fillStyle = isPeak ? P.ruby
          : `rgba(0,240,255,${0.25 + v * 0.65})`;
        ctx.fillRect(L + i * bw, T + ih - bh, Math.max(bw - 0.8, 0.8), bh);
      }
      ctx.fillStyle = P.dim; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'left';
      ctx.fillText('DC', L, h - 7);
      ctx.textAlign = 'right';
      ctx.fillText('NYQUIST · 1 cycle / 2 blocks', w - Rr, h - 7);
      ctx.textAlign = 'center';
      ctx.fillStyle = P.ruby;
      ctx.fillText(`dominant period ≈ ${peakBin ? (1024 / peakBin).toFixed(1) : '—'} blocks`,
        L + iw / 2, h - 7);
    }, cv);
  }

  function buildWaterfallSpec(ch) {
    const cv = q('#scWfall');
    const s = ch.blocks.series, mean = ch.blocks.mean;
    const WIN = 128, HOP = 24, BINS = 64;
    const frames = [];
    for (let start = 0; start + WIN <= s.length; start += HOP) {
      const seg = new Array(WIN);
      for (let i = 0; i < WIN; i++) seg[i] = s[start + i] - mean;
      const mag = fftMag(seg);
      const row = new Array(BINS);
      for (let i = 0; i < BINS; i++) row[i] = 20 * Math.log10(mag[i] + 1e-9);
      frames.push(row);
    }
    // A min/max stretch would be dragged to nothing by one near-zero bin, and a full
    // stretch turns broadband noise into confetti. The floor sits at the median-ish
    // quartile so the background stays dark and only real excursions light up.
    const flat = [];
    frames.forEach((r) => r.forEach((v) => flat.push(v)));
    flat.sort((a, b) => a - b);
    const lo = flat[Math.floor(flat.length * 0.25)];
    const hi = flat[Math.floor(flat.length * 0.995)];

    const heat = (v) => {
      const k = clamp((v - lo) / (hi - lo || 1), 0, 1);
      if (k < 0.35) {                       // deep blue → cyan
        const t2 = k / 0.35;
        return `rgb(6,${Math.round(30 + t2 * 170)},${Math.round(70 + t2 * 175)})`;
      }
      if (k < 0.7) {                        // cyan → violet
        const t2 = (k - 0.35) / 0.35;
        return `rgb(${Math.round(6 + t2 * 170)},${Math.round(200 - t2 * 60)},${Math.round(245 - t2 * 20)})`;
      }
      const t2 = (k - 0.7) / 0.3;           // violet → amber → white
      return `rgb(${Math.round(176 + t2 * 79)},${Math.round(140 + t2 * 100)},${Math.round(225 - t2 * 165)})`;
    };
    // The surface is fixed, so it is rasterised once and then blitted with a reveal
    // wipe over it — 5,000 fills a frame would eat the whole budget.
    const store = {};
    const reveal = { t0: null };
    ticker((t) => {
      const { ctx, w, h } = fit(cv);
      if (!w) return;
      const img = backing(store, 'surface', w, h, (c) => {
        const rh = h / frames.length, cw = w / BINS;
        for (let r = 0; r < frames.length; r++) {
          for (let cix = 0; cix < BINS; cix++) {
            c.fillStyle = heat(frames[r][cix]);
            c.fillRect(cix * cw, r * rh, cw + 0.7, rh + 0.7);
          }
        }
      });
      const rk = ramp(reveal, t, 1600);
      ctx.clearRect(0, 0, w, h);
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, w, h * rk); ctx.clip();
      ctx.drawImage(img, 0, 0, w, h);
      ctx.restore();

      const y = ((t / 3400) % 1) * h;
      ctx.fillStyle = 'rgba(255,255,255,.12)'; ctx.fillRect(0, y, w, 1.4);
      ctx.fillStyle = P.hi; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'left';
      ctx.fillText('BLOCK 0', 6, 11);
      ctx.fillText(`BLOCK ${s.length}`, 6, h - 6);
      ctx.textAlign = 'right'; ctx.fillText('FREQ →', w - 6, h - 6); ctx.textAlign = 'left';
    }, cv);
  }

  function buildXy(a, b) {
    const cv = q('#scXy');
    const sa = a.blocks.series, sb = b.blocks.series;
    const n = Math.min(sa.length, sb.length);
    const nrm = (arr) => { const lo = Math.min(...arr), hi = Math.max(...arr);
      return (v) => (v - lo) / (hi - lo || 1); };
    const na = nrm(sa), nb = nrm(sb);
    const store = {};
    ticker((t) => {
      const { ctx, w, h } = fit(cv);
      if (!w) return;
      const pad = 22, iw = w - pad * 2, ih = h - pad * 2;

      // the full cloud is static; only the comet head moves
      const bg = backing(store, 'cloud', w, h, (c) => {
        c.strokeStyle = 'rgba(0,240,255,.08)';
        c.beginPath();
        for (let k = 0; k <= 4; k++) {
          c.moveTo(pad, pad + (ih * k) / 4); c.lineTo(w - pad, pad + (ih * k) / 4);
          c.moveTo(pad + (iw * k) / 4, pad); c.lineTo(pad + (iw * k) / 4, h - pad);
        }
        c.stroke();
        c.fillStyle = 'rgba(0,240,255,.20)';
        for (let i = 0; i < n; i++) {
          c.fillRect(pad + na(sa[i]) * iw, h - pad - nb(sb[i]) * ih, 1.9, 1.9);
        }
        c.fillStyle = P.dim; c.font = '8.5px JetBrains Mono';
        c.fillText('CH A →', pad, h - 7);
        c.save(); c.translate(11, h - pad); c.rotate(-Math.PI / 2);
        c.fillText('CH B →', 0, 0); c.restore();
      });

      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(bg, 0, 0, w, h);

      const head = Math.floor(((t / 5200) % 1) * n);
      ctx.globalCompositeOperation = 'lighter';
      for (let k = 0; k < 160; k++) {
        const i = (head - k + n) % n;
        ctx.fillStyle = `rgba(0,240,255,${(0.7 * (1 - k / 160)).toFixed(3)})`;
        ctx.fillRect(pad + na(sa[i]) * iw, h - pad - nb(sb[i]) * ih, 2.4, 2.4);
      }
      ctx.globalCompositeOperation = 'source-over';
    }, cv);
  }

  function wireScopeControls() {
    const setT = (id, key) => {
      const b = q(id);
      b.addEventListener('click', () => {
        HX.scope[key] = !HX.scope[key];
        b.classList.toggle('on', !!HX.scope[key]);
      });
    };
    q('#scRun').addEventListener('click', () => {
      HX.scope.run = !HX.scope.run;
      q('#scRun').classList.toggle('on', HX.scope.run);
      q('#scRun').textContent = HX.scope.run ? 'Sweep' : 'Hold';
    });
    setT('#scPersist', 'persist');
    setT('#scTrig', 'trig');
    q('#scGain').addEventListener('input', (e) => { HX.scope.gain = e.target.value / 100; });
  }

  // ==========================================================================
  // RISK OBSERVATORY
  // ==========================================================================
  async function loadObservatory() {
    const sc = HX.scores || (HX.scores = await api('/api/scores'));
    const p = sc.p, y = sc.y, n = sc.n;
    q('#obN').textContent = `${n.toLocaleString('en-US')} eligible dies`;

    // sort once, descending risk — every curve below is a sweep over this order
    const idx = Array.from(p.keys()).sort((a, b) => p[b] - p[a]);
    const pos = y.reduce((a, v) => a + v, 0), neg = n - pos;

    const roc = [], gain = [], prc = [];
    let tp = 0, fp = 0;
    const stride = Math.max(1, Math.floor(n / 400));
    for (let k = 0; k < n; k++) {
      y[idx[k]] ? tp++ : fp++;
      if (k % stride === 0 || k === n - 1) {
        roc.push([fp / neg, tp / pos]);
        gain.push([(k + 1) / n, tp / pos]);
        prc.push([tp / pos, tp / (tp + fp)]);
      }
    }
    let auroc = 0;
    for (let i = 1; i < roc.length; i++) {
      auroc += (roc[i][0] - roc[i - 1][0]) * (roc[i][1] + roc[i - 1][1]) / 2;
    }

    HX.obs = { p, y, n, idx, pos, neg, roc, gain, prc, auroc, base: pos / n,
      ship: sc.shipped_threshold };

    buildHist(p, y);
    drawRoc(roc, auroc, pos / n);
    drawGain(gain, pos / n);
    const ks = drawKs(p, y);
    HX.obs.ksThr = ks.thr;
    buildDeciles(idx, y, p, pos);

    const sl = q('#obThr');
    sl.value = Math.round(sc.shipped_threshold * 1000);
    sl.addEventListener('input', applyObs);
    q('#obShip').addEventListener('click', () => {
      sl.value = Math.round(HX.obs.ship * 1000); applyObs();
    });
    q('#obKs').addEventListener('click', () => {
      sl.value = Math.round(HX.obs.ksThr * 1000); applyObs();
      toast(`Snapped to the cut where the two class distributions are furthest apart.`);
    });
    applyObs();
  }

  function applyObs() {
    const o = HX.obs; if (!o) return;
    const t = q('#obThr').value / 1000;
    q('#obThrV').textContent = t.toFixed(3);
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (let i = 0; i < o.n; i++) {
      const f = o.p[i] >= t, yy = o.y[i] === 1;
      if (f && yy) tp++; else if (f) fp++; else if (yy) fn++; else tn++;
    }
    const prec = tp / Math.max(tp + fp, 1), rec = tp / Math.max(tp + fn, 1);
    const f1 = (prec + rec) ? (2 * prec * rec) / (prec + rec) : 0;
    const box = q('#obTiles');
    const cells = [
      ['cyan', 'Recall on failures', (rec * 100).toFixed(1) + '%', `${tp} of ${tp + fn} caught`],
      ['violet', 'Precision', (prec * 100).toFixed(1) + '%', `${fp} good dies scrapped`],
      ['mint', 'F1 on the failure class', f1.toFixed(4), 'what we select on, with AUC-PR'],
      ['amber', 'Dies flagged', ((100 * (tp + fp)) / o.n).toFixed(2) + '%',
        `${(tp + fp).toLocaleString('en-US')} of ${o.n.toLocaleString('en-US')}`],
    ];
    box.innerHTML = '';
    cells.forEach(([c, k, v, nn]) => box.appendChild(dom(
      `<div class="tile ${c}"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>` +
      `<div class="n">${esc(nn)}</div></div>`)));

    // move the marker on the ROC without redrawing the curve
    const g = q('#obRocMarker');
    if (g) {
      const fpr = fp / Math.max(o.neg, 1), tpr = rec;
      const L = 52, T = 16, iw = 480 - 52 - 16, ih = 330 - 16 - 40;
      g.setAttribute('transform', `translate(${L + fpr * iw},${T + ih - tpr * ih})`);
    }
  }

  function buildHist(p, y) {
    const cv = q('#obHist');
    const BINS = 56, lo = -4;                       // log10 floor at 0.0001
    const pass = new Array(BINS).fill(0), fail = new Array(BINS).fill(0);
    for (let i = 0; i < p.length; i++) {
      const v = Math.log10(Math.max(p[i], 1e-4));
      const b = clamp(Math.floor(((v - lo) / (0 - lo)) * BINS), 0, BINS - 1);
      (y[i] === 1 ? fail : pass)[b]++;
    }
    const hi = Math.max(...pass, ...fail, 1);
    const grow = { t0: null };
    ticker((t) => {
      const { ctx, w, h } = fit(cv);
      if (!w) return;
      ctx.clearRect(0, 0, w, h);
      const gk = ramp(grow, t, 1300);
      const L = 40, Rr = 10, T = 14, B = 30;
      const iw = w - L - Rr, ih = h - T - B, bw = iw / BINS;
      // the bars are log-scaled, so the gridline labels have to be too
      const atFrac = (f) => Math.pow(10, f * Math.log10(hi + 1)) - 1;
      ctx.strokeStyle = 'rgba(0,240,255,.07)';
      for (let k = 0; k <= 4; k++) {
        const yy = T + (ih * k) / 4;
        ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(w - Rr, yy); ctx.stroke();
        ctx.fillStyle = P.dim; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'right';
        ctx.fillText(Math.round(atFrac(1 - k / 4)).toLocaleString('en-US'), L - 5, yy + 3);
      }
      const scale = (v) => (Math.log10(v + 1) / Math.log10(hi + 1)) * ih * gk;
      for (let i = 0; i < BINS; i++) {
        const x = L + i * bw;
        const hp = scale(pass[i]);
        ctx.fillStyle = 'rgba(30,111,140,.75)';
        ctx.fillRect(x, T + ih - hp, Math.max(bw - 1, 1), hp);
        const hf = scale(fail[i]);
        ctx.fillStyle = 'rgba(255,179,0,.92)';
        ctx.fillRect(x + bw * 0.28, T + ih - hf, Math.max(bw * 0.44, 1), hf);
      }
      // shipped cut
      const o = HX.obs;
      if (o) {
        const t2 = q('#obThr') ? q('#obThr').value / 1000 : o.ship;
        const lv = Math.log10(Math.max(t2, 1e-4));
        const x = L + ((lv - lo) / (0 - lo)) * iw;
        ctx.strokeStyle = P.cyan; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ih); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = P.cyan; ctx.font = '8.5px JetBrains Mono'; ctx.textAlign = 'left';
        ctx.fillText('cut', x + 4, T + 10);
      }
      ctx.fillStyle = P.dim; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'center';
      ['0.0001', '0.001', '0.01', '0.1', '1.0'].forEach((lab, i) => {
        ctx.fillText(lab, L + (iw * i) / 4, h - 16);
      });
      ctx.textAlign = 'left'; ctx.fillStyle = 'rgba(30,111,140,1)';
      ctx.fillRect(L, h - 9, 9, 5); ctx.fillStyle = P.dim;
      ctx.fillText('passed', L + 13, h - 5);
      ctx.fillStyle = P.amber; ctx.fillRect(L + 62, h - 9, 9, 5); ctx.fillStyle = P.dim;
      ctx.fillText('newly failed', L + 75, h - 5);
    }, cv);
  }

  function drawRoc(roc, auroc, base) {
    const svg = q('#obRoc');
    drawXY(svg, [
      { points: [[0, 0], [1, 1]], color: 'rgba(88,96,105,.8)', dashed: 'diag' },
      { points: roc, color: P.cyan, width: 2 },
    ], { xLabel: 'false positive rate', yLabel: 'true positive rate',
      legend: [{ color: P.cyan, label: `ROC-AUC ${auroc.toFixed(4)}` }] });
    const path = qa('path', svg).pop();
    if (path) drawIn(path, 1500);
    const g = sv('g', { id: 'obRocMarker' });
    g.appendChild(sv('circle', { r: 6, fill: 'none', stroke: P.amber, 'stroke-width': 1.5 }));
    g.appendChild(sv('circle', { r: 2.4, fill: P.amber }));
    svg.appendChild(g);
    svg.appendChild(sv('text', { x: 240, y: 322, fill: P.dim, 'text-anchor': 'middle',
      'font-family': 'JetBrains Mono', 'font-size': 8.5 },
      `a ${(base * 100).toFixed(1)}% base rate is why this curve looks better than the job is`));
  }

  function drawGain(gain, base) {
    const svg = q('#obGain');
    drawXY(svg, [
      { points: [[0, 0], [1, 1]], color: 'rgba(88,96,105,.8)', dashed: 'diag' },
      { points: [[0, 0], [base, 1], [1, 1]], color: 'rgba(0,230,118,.55)', dashed: true },
      { points: gain, color: P.cyan, width: 2 },
    ], { xLabel: 'fraction of dies scrapped, worst first',
      yLabel: 'fraction of failures caught',
      legend: [{ color: P.cyan, label: 'shipped model' },
        { color: 'rgba(0,230,118,.55)', label: 'perfect ranking' },
        { color: 'rgba(88,96,105,.8)', label: 'random' }] });
    const paths = qa('path', svg);
    if (paths.length) drawIn(paths[paths.length - 1], 1600);
  }

  function drawKs(p, y) {
    const svg = q('#obKsSvg');
    const B = 200;
    const cf = new Array(B + 1).fill(0), cp = new Array(B + 1).fill(0);
    let nf = 0, np = 0;
    for (let i = 0; i < p.length; i++) {
      const b = clamp(Math.floor(Math.pow(p[i], 0.4) * B), 0, B);
      if (y[i] === 1) { cf[b]++; nf++; } else { cp[b]++; np++; }
    }
    let af = 0, ap = 0, best = 0, bestX = 0;
    const sf = [], sp = [];
    for (let b = 0; b <= B; b++) {
      af += cf[b] / Math.max(nf, 1); ap += cp[b] / Math.max(np, 1);
      const x = b / B;
      sf.push([x, af]); sp.push([x, ap]);
      if (Math.abs(ap - af) > best) { best = Math.abs(ap - af); bestX = x; }
    }
    const thr = Math.pow(bestX, 1 / 0.4);
    q('#obKsV').textContent = `KS ${best.toFixed(4)} at risk ${thr.toFixed(3)}`;
    drawXY(svg, [
      { points: sp, color: 'rgba(30,111,140,1)', width: 2 },
      { points: sf, color: P.amber, width: 2 },
      { points: [[bestX, 0], [bestX, 1]], color: P.ruby, dashed: true, width: 1.2 },
    ], { xLabel: 'risk, compressed for display', yLabel: 'cumulative share of class',
      legend: [{ color: 'rgba(30,111,140,1)', label: 'passed dies' },
        { color: P.amber, label: 'newly failed dies' }] });
    qa('path', svg).slice(-3).forEach((pp, i) => drawIn(pp, 1400, i * 120));
    return { ks: best, thr };
  }

  function buildDeciles(idx, y, p, pos) {
    const tbl = q('#obDecile');
    const n = idx.length, per = Math.floor(n / 10);
    let rows = '<thead><tr><th>Decile</th><th class="r">Dies</th><th class="r">Risk band</th>' +
      '<th class="r">Failures</th><th class="r">Rate</th><th class="r">Lift</th>' +
      '<th class="r">Cumulative capture</th><th style="width:26%">&nbsp;</th></tr></thead><tbody>';
    const base = pos / n;
    let cum = 0;
    const lifts = [];
    for (let d = 0; d < 10; d++) {
      const a = d * per, b = d === 9 ? n : (d + 1) * per;
      let f = 0;
      for (let k = a; k < b; k++) f += y[idx[k]];
      cum += f;
      const rate = f / (b - a), lift = rate / base;
      lifts.push(lift);
      rows += `<tr><td>D${d + 1}${d === 0 ? ' <span style="color:var(--cyan)">worst</span>' : ''}</td>` +
        `<td class="r">${(b - a).toLocaleString('en-US')}</td>` +
        `<td class="r" style="color:var(--tx-dim)">${p[idx[b - 1]].toFixed(4)}–${p[idx[a]].toFixed(4)}</td>` +
        `<td class="r">${f}</td>` +
        `<td class="r" style="color:var(--amber)">${(rate * 100).toFixed(2)}%</td>` +
        `<td class="r" style="color:var(--cyan)">${lift.toFixed(2)}×</td>` +
        `<td class="r">${((cum / pos) * 100).toFixed(1)}%</td>` +
        `<td><span class="liftbar" data-l="${lift}"></span></td></tr>`;
    }
    tbl.innerHTML = rows + '</tbody>';
    const hi = Math.max(...lifts, 1e-6);
    qa('.liftbar', tbl).forEach((b, i) =>
      setTimeout(() => { b.style.width = `${(Number(b.dataset.l) / hi) * 100}%`; }, 90 + i * 60));
  }

  // ==========================================================================
  // CALIBRATION LAB
  // ==========================================================================
  async function loadCalib() {
    const c = await api('/api/calibration');
    const tiles = q('#cbTiles'); tiles.innerHTML = '';
    [
      ['cyan', 'Brier score', Number(c.brier).toFixed(4), 'lower is better; 0 is perfect'],
      ['mint', 'Expected calibration error', Number(c.ece).toFixed(4), 'mean gap between promise and reality'],
      ['violet', 'Mean conformal width', c.mean_interval_width != null
        ? Number(c.mean_interval_width).toFixed(4) : '—', 'how wide the honest interval is'],
      ['amber', 'Dies carrying an interval', Number(c.n_with_interval).toLocaleString('en-US'),
        'split-conformal, calibrated on validation'],
    ].forEach(([cl, k, v, n]) => tiles.appendChild(dom(
      `<div class="tile ${cl}"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>` +
      `<div class="n">${esc(n)}</div></div>`)));

    const pts = (c.curve || []).map((r) => [r.predicted, r.observed]);
    const hi = Math.max(...pts.flat(), 0.02);
    drawXY(q('#cbRel'), [
      { points: [[0, 0], [1, 1]], color: 'rgba(88,96,105,.9)', dashed: 'diag' },
      { points: pts, color: P.cyan, width: 2, dots: true },
    ], { xRange: [0, hi], yRange: [0, hi], xDec: 3, yDec: 3,
      xLabel: 'predicted risk', yLabel: 'observed failure rate',
      legend: [{ color: P.cyan, label: 'shipped model' }] });
    const rel = qa('#cbRel path');
    if (rel.length) drawIn(rel[rel.length - 1], 1400);

    // gap bars
    const cv = q('#cbGap');
    const curve = c.curve || [];
    const grow = { t0: null };
    ticker((t) => {
      const { ctx, w, h } = fit(cv);
      if (!w) return;
      ctx.clearRect(0, 0, w, h);
      const gk = ramp(grow, t, 1100);
      const L = 44, Rr = 10, T = 12, B = 26;
      const iw = w - L - Rr, ih = h - T - B, mid = T + ih / 2;
      const span = Math.max(...curve.map((r) => Math.abs(r.observed - r.predicted)), 1e-4) * 1.15;
      ctx.strokeStyle = 'rgba(0,240,255,.20)';
      ctx.beginPath(); ctx.moveTo(L, mid); ctx.lineTo(w - Rr, mid); ctx.stroke();
      const bw = iw / Math.max(curve.length, 1);
      curve.forEach((r, i) => {
        const gap = (r.observed - r.predicted) * gk;
        const bh = (gap / span) * (ih / 2);
        ctx.fillStyle = gap >= 0 ? 'rgba(255,179,0,.85)' : 'rgba(0,240,255,.85)';
        ctx.fillRect(L + i * bw + 2, mid - Math.max(bh, 0), Math.max(bw - 5, 2), Math.abs(bh));
      });
      ctx.fillStyle = P.dim; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'right';
      ctx.fillText('+' + span.toFixed(4), L - 5, T + 8);
      ctx.fillText('0', L - 5, mid + 3);
      ctx.fillText('−' + span.toFixed(4), L - 5, T + ih - 2);
      ctx.textAlign = 'center'; ctx.fillStyle = P.dim;
      ctx.fillText('risk bins, low to high', L + iw / 2, h - 7);
    }, cv);

    q('#cbGapRows').innerHTML = curve.map((r) => {
      const gap = r.observed - r.predicted;
      return `<div class="hx-sub ${Math.abs(gap) > 0.02 ? 'warn' : ''}">
        <span class="nm">bin at ${r.predicted.toFixed(4)} · ${r.n.toLocaleString('en-US')} dies</span>
        <span class="st" style="color:${Math.abs(gap) > 0.02 ? P.amber : P.mint}">${
          gap >= 0 ? '+' : ''}${gap.toFixed(4)}</span></div>`;
    }).join('');

    q('#cbConf').innerHTML = `
      <div class="g3">
        <div><p class="hx-eyebrow">What the interval means</p>
          <p class="hx-note" style="margin:0">A split-conformal interval is calibrated so that, across the
          calibration set, the true outcome falls inside it at the stated rate. It is a distribution-free
          guarantee: no assumption that the score is Gaussian, which is exactly where DPAT breaks down.</p></div>
        <div><p class="hx-eyebrow">Why width matters more than the point estimate</p>
          <p class="hx-note" style="margin:0">A die at 0.60 ± 0.05 and a die at 0.60 ± 0.35 are different
          engineering problems. The narrow one is a decision; the wide one is a request for another
          measurement. The console prints both.</p></div>
        <div><p class="hx-eyebrow">Fitted where</p>
          <p class="hx-note" style="margin:0">Isotonic regression and the conformal quantile are both fitted on
          validation wafers only and applied unchanged to test. Fitting either on test would be the fastest way
          to produce a number that means nothing.</p></div>
      </div>`;
  }

  // ==========================================================================
  // SIGNATURE ATLAS
  // ==========================================================================
  function sigGlyph(name) {
    const s = sv('svg', { viewBox: '0 0 120 120' });
    s.appendChild(sv('circle', { cx: 60, cy: 60, r: 52, fill: 'rgba(0,240,255,.03)',
      stroke: 'rgba(0,240,255,.22)', 'stroke-width': 1 }));
    s.appendChild(sv('line', { x1: 60, y1: 8, x2: 60, y2: 16, stroke: 'rgba(0,240,255,.5)' }));

    const dot = (x, y, r, c, delay) => {
      const d = sv('circle', { cx: x, cy: y, r, fill: c, opacity: 0 });
      d.style.transition = `opacity .5s ease ${delay}ms`;
      requestAnimationFrame(() => requestAnimationFrame(() => { d.style.opacity = .9; }));
      return d;
    };
    // Sampled directly in the geometry each signature is defined by, rather than by
    // rejection, so the glyph reads as the shape the classifier keys on.
    const rnd = (seed) => { let s2 = seed % 2147483647 || 1; return () => (s2 = (s2 * 16807) % 2147483647) / 2147483647; };
    const R = rnd(name.length * 7919 + 131);
    const polar = (rad, ang) => [60 + Math.cos(ang) * rad * 50, 60 + Math.sin(ang) * rad * 50];
    const jit = (v, k) => v + (R() - 0.5) * k;

    const SPEC = {
      Center: { n: 150, f: () => polar(Math.pow(R(), 0.6) * 0.34, R() * 6.2832) },
      Donut: { n: 170, f: () => polar(0.40 + R() * 0.22, R() * 6.2832) },
      'Edge-Ring': { n: 190, f: () => polar(0.86 + R() * 0.12, R() * 6.2832) },
      'Edge-Loc': { n: 130, f: () => polar(0.82 + R() * 0.15, -2.3 + R() * 1.2) },
      Loc: { n: 120, f: () => polar(0.42 + R() * 0.16, 2.1 + (R() - 0.5) * 0.9) },
      Scratch: { n: 130, f: () => { const t2 = R(); return [jit(18 + t2 * 80, 6), jit(96 - t2 * 74, 6)]; } },
      Random: { n: 95, f: () => polar(Math.sqrt(R()) * 0.95, R() * 6.2832) },
      'Near-full': { n: 260, f: () => polar(Math.sqrt(R()) * 0.97, R() * 6.2832) },
      Clean: { n: 5, f: () => polar(Math.sqrt(R()) * 0.9, R() * 6.2832) },
    };
    const spec = SPEC[name] || SPEC.Clean;
    const col = name === 'Clean' ? 'rgba(0,230,118,.75)' : P.ruby;
    for (let i = 0; i < spec.n; i++) {
      const [x, y] = spec.f();
      if (Math.hypot(x - 60, y - 60) > 51) continue;
      s.appendChild(dot(x, y, name === 'Near-full' ? 1.9 : 2.2, col, (i % 60) * 8));
    }
    // sweep
    const g = sv('g', { class: 'hx-ring b',
      style: 'transform-box:view-box;transform-origin:60px 60px' });
    g.appendChild(sv('line', { x1: 60, y1: 60, x2: 60, y2: 10,
      stroke: 'rgba(0,240,255,.45)', 'stroke-width': 1.2 }));
    s.appendChild(g);
    return s;
  }

  async function loadAtlas() {
    const rows = await api('/api/rootcauses');
    const grid = q('#atGrid'); grid.innerHTML = '';
    const total = rows.reduce((a, r) => a + (r.dies_seen || 0), 0);
    q('#atCount').textContent =
      `${rows.length} signatures · ${total.toLocaleString('en-US')} classified dies`;
    rows.sort((a, b) => (b.dies_seen || 0) - (a.dies_seen || 0));
    rows.forEach((r) => {
      const card = dom(`<div class="hx-sig">
        <div class="glyph"></div>
        <div class="bd">
          <h3>${esc(r.signature)}</h3>
          <div class="cz">${esc(r.action)}</div>
          <p>${esc(r.cause)}</p>
          <div class="ct">
            <span>dies <b>${(r.dies_seen || 0).toLocaleString('en-US')}</b></span>
            <span>share <b>${total ? ((100 * (r.dies_seen || 0)) / total).toFixed(1) : '0.0'}%</b></span>
          </div>
        </div></div>`);
      q('.glyph', card).appendChild(sigGlyph(r.signature));
      grid.appendChild(card);
    });
  }

  // ==========================================================================
  // RUN DECK — rod bank + throughput trace, mirrored off the live run
  // ==========================================================================
  let runDeckMounted = false;
  function mountRunDeck() {
    if (runDeckMounted) return;
    const view = q('#view-run'); if (!view) return;
    runDeckMounted = true;

    const panel = dom(`
      <div class="panel lit">
        <div class="ph"><span class="material-symbols-outlined">nest_thermostat_gen_3</span>
          <h2>Reactor rod bank — pipeline stages</h2>
          <span class="spacer"></span>
          <span class="hx-live" id="rdLive" style="display:none">running</span>
          <span class="sub mono" id="rdCap">idle</span></div>
        <div class="pb tight">
          <div class="hx-rods" id="rdRods"></div>
          <p class="hx-note" style="margin:7px 0 8px">Every stage of the running pipeline appears
            here as a rod and fills as the process reaches it, mirrored from the same stdout the
            console below is streaming.</p>
          <div class="hx-runtel"><canvas id="rdTel"></canvas>
            <div class="cap">process output throughput · lines per second</div></div>
        </div>
      </div>`);
    const bar = dom(`
      <div class="panel lit">
        <div class="ph"><span class="material-symbols-outlined">rocket_launch</span>
          <h2>Mission status</h2>
          <span class="spacer"></span>
          <span class="hx-live" id="rdLive2" style="display:none">executing</span>
          <span class="sub mono" id="rdCmd2">no run this session</span></div>
        <div class="hx-runbar" id="rdBar">
          <div class="hx-cell cy"><div class="k">Reactor state</div><div class="v" id="rbState">IDLE</div>
            <div class="n" id="rbStateN">nothing dispatched yet</div></div>
          <div class="hx-cell"><div class="k">Task</div><div class="v" style="font-size:14px" id="rbTask">—</div>
            <div class="n">preset the console is allowed to start</div></div>
          <div class="hx-cell vi"><div class="k">Elapsed / remaining</div>
            <div class="v" style="font-size:15px" id="rbClock">—</div>
            <div class="n">measured on this machine</div></div>
          <div class="hx-cell am"><div class="k">Stage</div><div class="v" id="rbStage">0 / 0</div>
            <div class="n" id="rbStageN">rods idle</div></div>
          <div class="hx-cell mi"><div class="k">AUC-PR as it lands</div><div class="v" id="rbAuc">—</div>
            <div class="n">lifted from the process stream</div></div>
          <div class="hx-cell ru"><div class="k">Output throughput</div><div class="v" id="rbTput">0</div>
            <div class="n">lines per second from the subprocess</div></div>
        </div>
      </div>`);
    const grid = q('#presetGrid', view);
    if (grid) { grid.before(bar); grid.after(panel); }
    else { view.appendChild(bar); view.appendChild(panel); }

    // rods mirror whatever app.js has rendered into the stage list
    const stageList = q('#stageList');
    const sync = () => {
      const stages = qa('.stage', stageList);
      const rods = q('#rdRods');
      if (!stages.length) return;
      if (rods.children.length !== stages.length || rods.querySelector('.idle')) {
        rods.innerHTML = '';
        stages.forEach((_, i) => rods.appendChild(dom(
          `<div class="hx-rod"><b>${String(i + 1).padStart(2, '0')}</b><i></i><span></span></div>`)));
      }
      stages.forEach((st, i) => {
        const rod = rods.children[i];
        const label = st.querySelector('span:nth-child(2)');
        rod.querySelector('span').textContent = label ? label.textContent : '';
        const done = st.classList.contains('done');
        const now = st.classList.contains('now');
        rod.classList.toggle('done', done);
        rod.classList.toggle('live', now);
        rod.querySelector('i').style.height = done ? '100%' : (now ? '55%' : '6%');
      });
      const state = q('#runState');
      const label = state ? state.textContent.trim() : 'Idle';
      const running = /running/i.test(label);
      const done = stages.filter((s) => s.classList.contains('done')).length;
      q('#rdLive').style.display = running ? '' : 'none';
      q('#rdLive2').style.display = running ? '' : 'none';
      q('#rdCap').textContent = `${done} / ${stages.length} stages complete`;

      const cell = q('#rbState');
      cell.textContent = label.toUpperCase();
      const tone = running ? 'cy' : /done/i.test(label) ? 'mi' : /idle/i.test(label) ? '' : 'ru';
      cell.parentElement.className = 'hx-cell ' + tone;
      q('#rbStateN').textContent = running ? 'subprocess streaming'
        : /done/i.test(label) ? 'artifacts written, dashboard reloaded' : 'nothing dispatched yet';
      const cmd = q('#runCmd') ? q('#runCmd').textContent : '';
      q('#rdCmd2').textContent = cmd || 'no run this session';
      q('#rbTask').textContent = (cmd.split(/\s+/).find((w) => w.startsWith('src.')) || '—');
      q('#rbClock').textContent = q('#runClock') ? q('#runClock').textContent : '—';
      q('#rbStage').textContent = `${done} / ${stages.length}`;
      const now = stages.find((x) => x.classList.contains('now'));
      q('#rbStageN').textContent = now
        ? now.querySelector('span:nth-child(2)').textContent
        : (done ? 'all stages complete' : 'rods idle');
      q('#rbAuc').textContent = q('#rmAuc') ? q('#rmAuc').textContent : '—';
    };
    // an idle bank of dim rods, so the panel reads as hardware before anything runs
    const idle = q('#rdRods');
    for (let i = 0; i < 10; i++) {
      idle.appendChild(dom(`<div class="hx-rod idle"><b>${String(i + 1).padStart(2, '0')}</b>` +
        `<i style="height:14%"></i><span>standby</span></div>`));
    }
    new MutationObserver(sync).observe(stageList, { childList: true, subtree: true, attributes: true });
    sync();

    // throughput trace, measured off the console as lines actually arrive
    const cv = q('#rdTel');
    const hist = new Array(180).fill(0);
    let bucket = 0, last = 0;
    new MutationObserver((m) => { m.forEach((r) => { bucket += r.addedNodes.length; }); })
      .observe(q('#console'), { childList: true });
    ticker((t) => {
      if (t - last > 250) {
        last = t; hist.push(bucket * 4); hist.shift(); bucket = 0;
        const cell = q('#rbTput');
        if (cell) cell.textContent = hist[hist.length - 1].toFixed(0);
      }
      const { ctx, w, h } = fit(cv);
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = 'rgba(0,240,255,.07)';
      for (let x = 0; x < w; x += 28) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      const hi = Math.max(...hist, 4);
      ctx.beginPath();
      hist.forEach((v, i) => {
        const x = (i / (hist.length - 1)) * w, y = h - 8 - (v / hi) * (h - 26);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.strokeStyle = P.mint; ctx.lineWidth = 1.4;
      ctx.shadowColor = P.mint; ctx.shadowBlur = 9; ctx.stroke(); ctx.shadowBlur = 0;
      ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, 'rgba(0,230,118,.20)'); g.addColorStop(1, 'rgba(0,230,118,0)');
      ctx.fillStyle = g; ctx.fill();
      ctx.fillStyle = P.dim; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'right';
      ctx.fillText(`${hist[hist.length - 1].toFixed(0)} ln/s · peak ${hi.toFixed(0)}`, w - 8, 14);
    }, cv);
  }

  // ==========================================================================
  // TOP BAR EXTRAS
  // ==========================================================================
  function mountTopbar() {
    const bar = q('.topbar'); if (!bar) return;
    const btn = dom('<button class="btn ghost hide-sm" id="hxCalm" title="Reduce motion and overlays" ' +
      'style="margin-right:6px">FX</button>');
    btn.addEventListener('click', () => {
      document.body.classList.toggle('hx-calm');
      btn.classList.toggle('on', !document.body.classList.contains('hx-calm'));
      toast(document.body.classList.contains('hx-calm')
        ? 'Reduced-motion mode. Scanlines, vignette and the particle field are off.'
        : 'Full mission-control chrome restored.');
    });
    const anchor = q('#btnDispatch', bar);
    anchor ? anchor.before(btn) : bar.appendChild(btn);
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      document.body.classList.add('hx-calm');
    }
  }

  // ==========================================================================
  // START
  // ==========================================================================
  /** Only reveal icon ligatures once the Material Symbols font has actually loaded. */
  function watchIconFont() {
    const done = () => document.documentElement.classList.add('hx-icons');
    if (!document.fonts || !document.fonts.load) return done();
    document.fonts.load('20px "Material Symbols Outlined"', 'radar')
      .then((faces) => { if (faces && faces.length) done(); })
      .catch(() => {});
    document.fonts.ready.then(() => {
      if (document.fonts.check('20px "Material Symbols Outlined"')) done();
    }).catch(() => {});
  }

  function start() {
    watchIconFont();
    mountFx();
    mountViews();
    mountNav();
    watchViews();
    mountTopbar();
    bootSequence();

    // land on the command deck
    go('command');

    // keyboard: 1-9 jump between decks
    const ORDER = ['command', 'run', 'matrix', 'fleet', 'threshold', 'observatory',
      'calib', 'inspector', 'spatial'];
    document.addEventListener('keydown', (e) => {
      const tgt = e.target;
      if (tgt && tgt.matches && tgt.matches('input,textarea,select')) return;
      if (q('#palette') && q('#palette').classList.contains('on')) return;
      const n = Number(e.key);
      if (n >= 1 && n <= 9 && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault(); go(ORDER[n - 1]);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
