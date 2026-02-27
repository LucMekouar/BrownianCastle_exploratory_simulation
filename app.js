// Brownian Castle / β-Ballistic Deposition simulator
// ---------------------------------------------------
// Range-R extension implemented as additional Poisson clocks:
//
// For each site x:
//   - a center “+1” clock proposes y0 = h(x)+1
//   - for each k=1..R:
//       right-k clock proposes y(+k) = g(h(x), h(x+k))
//       left-k  clock proposes y(-k) = g(h(x), h(x-k))
//
// Averaging toggle affects only neighbour clocks:
//   - default: g(a,b)=b (copy neighbour height)
//   - averaging ON: g(a,b)=round((a+b)/2)
// The +1 clock always does h(x) <- h(x)+1.
//
// Choice among candidates uses softmax:
//    P(i) ∝ r_i * exp(β y_i)
// with paper normalisation giving r_center=2, r_neighbour=1.

const $ = (sel) => /** @type {HTMLElement} */ (document.querySelector(sel));

const clamp01 = (t) => Math.max(0, Math.min(1, t));
const lerp = (a, b, t) => a + (b - a) * t;

/** Piecewise linear gradient: green -> purple -> blue (castle palette). */
function castleRGB(t) {
  t = clamp01(t);
  const c0 = [72, 170, 102];
  const c1 = [176, 102, 204];
  const c2 = [96, 142, 230];
  let a, b, u;
  if (t < 0.55) { a = c0; b = c1; u = t / 0.55; }
  else { a = c1; b = c2; u = (t - 0.55) / 0.45; }
  return [
    Math.round(lerp(a[0], b[0], u)),
    Math.round(lerp(a[1], b[1], u)),
    Math.round(lerp(a[2], b[2], u)),
  ];
}

// --------- Deterministic PRNG ----------
function xmur3(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return function() {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return h >>> 0;
  };
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class RNG {
  /** @param {string} seedStr */
  constructor(seedStr) {
    const h = xmur3(seedStr);
    this._rand = mulberry32(h());
  }
  next() { return this._rand(); }
  int(n) { return (this._rand() * n) | 0; }
}

// --------- Event ring buffer for Brick view ----------
class EventRing {
  /** @param {number} capacity */
  constructor(capacity) { this.resize(capacity); }

  /** @param {number} capacity */
  resize(capacity) {
    this.capacity = Math.max(1, capacity | 0);
    this.x = new Uint32Array(this.capacity);
    this.y = new Int32Array(this.capacity);
    this.idx = new Uint32Array(this.capacity);
    this.head = 0;
    this.size = 0;
  }

  /** @param {number} x @param {number} y @param {number} idx */
  push(x, y, idx) {
    const i = this.head;
    this.x[i] = x >>> 0;
    this.y[i] = y | 0;
    this.idx[i] = idx >>> 0;
    this.head = (this.head + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }

  *iter() {
    for (let k = 0; k < this.size; k++) {
      const i = (this.head - 1 - k + this.capacity) % this.capacity;
      yield { x: this.x[i], y: this.y[i], idx: this.idx[i] };
    }
  }
}

// --------- Model ----------
class BetaBD {
  /** @param {number} N @param {RNG} rng */
  constructor(N, rng) {
    this.setSize(N);
    this.rng = rng;
    this.eventCount = 0;
    this.microTime = 0;
  }

  /** @param {number} N */
  setSize(N) {
    this.N = Math.max(4, N | 0);
    this.h = new Int32Array(this.N);
  }

  reset() {
    this.h.fill(0);
    this.eventCount = 0;
    this.microTime = 0;
  }

  /**
   * One event at uniformly random x.
   * @returns {{x:number, yNew:number}}
   */
  stepEvent(beta, betaInf, rNeighbour, rCenter, R, avgOn) {
    const N = this.N;
    const x = this.rng.int(N);
    const idx = (i) => (i + N) % N;

    const hx = this.h[x];

    // Candidates: 1 + 2R
    const nCand = 1 + 2 * R;
    const y = new Float64Array(nCand);
    const r = new Float64Array(nCand);

    // center +1
    y[0] = hx + 1;
    r[0] = rCenter;

    // neighbours ±k
    let j = 1;
    for (let k = 1; k <= R; k++) {
      const hr = this.h[idx(x + k)];
      const hl = this.h[idx(x - k)];

      y[j] = avgOn ? 0.5 * (hx + hr) : hr;
      r[j] = rNeighbour;
      j++;

      y[j] = avgOn ? 0.5 * (hx + hl) : hl;
      r[j] = rNeighbour;
      j++;
    }

    // choose index
    let choice = 0;

    if (betaInf) {
      let m = y[0];
      for (let i = 1; i < nCand; i++) if (y[i] > m) m = y[i];
      const cands = [];
      for (let i = 0; i < nCand; i++) if (y[i] === m) cands.push(i);
      choice = cands[this.rng.int(cands.length)];
    } else {
      let m = y[0];
      for (let i = 1; i < nCand; i++) if (y[i] > m) m = y[i];

      let sum = 0;
      const w = new Float64Array(nCand);
      for (let i = 0; i < nCand; i++) {
        const wi = r[i] * Math.exp(beta * (y[i] - m));
        w[i] = wi;
        sum += wi;
      }

      const u = this.rng.next() * sum;
      let acc = 0;
      for (let i = 0; i < nCand; i++) {
        acc += w[i];
        if (u <= acc) { choice = i; break; }
      }
    }

    // apply update (round only matters for averaging; +1 stays integer)
    const yNew = (choice === 0) ? (hx + 1) : Math.round(y[choice]);
    this.h[x] = yNew;
    this.eventCount++;

    return { x, yNew };
  }

  *batch(nEvents, beta, betaInf, rNeighbour, rCenter, R, avgOn, exactClocks) {
    const N = this.N;
    for (let i = 0; i < nEvents; i++) {
      const ev = this.stepEvent(beta, betaInf, rNeighbour, rCenter, R, avgOn);
      if (exactClocks) {
        const u = Math.max(1e-12, this.rng.next());
        this.microTime += -Math.log(u) / N;
      } else {
        this.microTime = this.eventCount / N; // sweeps
      }
      yield ev;
    }
  }

  stats() {
    const h = this.h;
    let min = h[0], max = h[0];
    let sum = 0;
    for (let i = 0; i < h.length; i++) {
      const v = h[i];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    return { min, max, mean: sum / h.length };
  }
}

// --------- Rendering ----------
class CanvasView {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    this.pixelRatio = Math.max(1, window.devicePixelRatio || 1);
    this.resizeToDisplaySize();
    window.addEventListener("resize", () => this.resizeToDisplaySize());
  }

  resizeToDisplaySize() {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(480, Math.floor(rect.width * this.pixelRatio));
    const h = Math.max(360, Math.floor((rect.width * 0.54) * this.pixelRatio));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  clear() {
    const ctx = this.ctx;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Draw last M events as points.
   * Returns the y-bounds used so the surface overlay can share coordinates.
   */
  drawBricks(eventsArray, opts) {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    if (!eventsArray || eventsArray.length === 0) return { yMin: -1, yMax: 1 };

    // bounds from events, respecting centering
    let yMin = eventsArray[0].y - (opts.center ? opts.centerValue : 0);
    let yMax = yMin;
    for (let i = 0; i < eventsArray.length; i++) {
      const v = eventsArray[i].y - (opts.center ? opts.centerValue : 0);
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
    if (yMax - yMin < 2) { yMax += 1; yMin -= 1; }
    const pad = 0.12 * (yMax - yMin);
    yMin -= pad; yMax += pad;

    const scaleY = opts.autoScale ? (H / (yMax - yMin)) : (opts.pxPerUnit * this.pixelRatio);
    const scaleX = W / opts.N;

    const newestIdx = eventsArray[0].idx;
    const oldestIdx = eventsArray[eventsArray.length - 1].idx;
    const span = Math.max(1, newestIdx - oldestIdx);

    ctx.save();
    for (let k = 0; k < eventsArray.length; k++) {
      const ev = eventsArray[k];
      const age01 = (newestIdx - ev.idx) / span;

      const x = (ev.x + 0.5) * scaleX;
      const yVal = ev.y - (opts.center ? opts.centerValue : 0);
      const y = H - (yVal - yMin) * scaleY;

      const rgb = castleRGB(1.0 - age01);
      const alpha = 0.20 + 0.65 * (1.0 - 0.35 * age01);
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;

      const s = Math.max(1, 1.6 * this.pixelRatio);
      ctx.fillRect(x - s / 2, y - s / 2, s, s);
    }
    ctx.restore();

    return { yMin, yMax };
  }

  /** Draw surface, optionally sharing y-bounds with bricks. */
  drawSurface(heights, opts) {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    const mean = opts.center ? opts.centerValue : 0;

    let yMin, yMax;
    if (opts.yMin != null && opts.yMax != null) {
      yMin = opts.yMin;
      yMax = opts.yMax;
    } else {
      let min = heights[0] - mean, max = heights[0] - mean;
      for (let i = 0; i < heights.length; i++) {
        const v = heights[i] - mean;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      yMin = min; yMax = max;
      if (yMax - yMin < 2) { yMax += 1; yMin -= 1; }
      const pad = 0.08 * (yMax - yMin);
      yMin -= pad; yMax += pad;
    }

    const scaleY = opts.autoScale ? (H / (yMax - yMin)) : (opts.pxPerUnit * this.pixelRatio);
    const scaleX = W / heights.length;

    ctx.save();
    ctx.lineWidth = Math.max(1, 1.5 * this.pixelRatio);
    ctx.strokeStyle = "rgba(96,142,230,0.95)";
    ctx.beginPath();
    for (let i = 0; i < heights.length; i++) {
      const x = (i + 0.5) * scaleX;
      const yVal = heights[i] - mean;
      const y = H - (yVal - yMin) * scaleY;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }
}

// --------- UI wiring ----------
const canvasEl = /** @type {HTMLCanvasElement} */ ($("#canvas"));
const view = new CanvasView(canvasEl);

const hudMode = $("#hudMode");
const hudEvents = $("#hudEvents");
const hudSweeps = $("#hudSweeps");
const hudMean = $("#hudMean");
const hudMinMax = $("#hudMinMax");
const hudFps = $("#hudFps");

const speed = /** @type {HTMLInputElement} */ ($("#speed"));
const speedVal = $("#speedVal");
const nSites = /** @type {HTMLInputElement} */ ($("#nSites"));
const nSitesVal = $("#nSitesVal");
const mBlocks = /** @type {HTMLInputElement} */ ($("#mBlocks"));
const mBlocksVal = $("#mBlocksVal");
const heightScale = /** @type {HTMLInputElement} */ ($("#heightScale"));
const heightScaleVal = $("#heightScaleVal");
const autoScale = /** @type {HTMLInputElement} */ ($("#autoScale"));

const beta = /** @type {HTMLInputElement} */ ($("#beta"));
const betaVal = $("#betaVal");
const betaInf = /** @type {HTMLInputElement} */ ($("#betaInf"));

const toggleBrick = /** @type {HTMLInputElement} */ ($("#toggleBrick"));
const toggleSurface = /** @type {HTMLInputElement} */ ($("#toggleSurface"));

const rangeR = /** @type {HTMLInputElement} */ ($("#rangeR"));
const rangeRVal = $("#rangeRVal");
const avgMode = /** @type {HTMLInputElement} */ ($("#avgMode"));

const usePaperRates = /** @type {HTMLInputElement} */ ($("#usePaperRates"));
const subtractDrift = /** @type {HTMLInputElement} */ ($("#subtractDrift"));
const seedInput = /** @type {HTMLInputElement} */ ($("#seed"));
const btnReseed = $("#btnReseed");
const exactClocks = /** @type {HTMLInputElement} */ ($("#exactClocks"));

const btnPause = $("#btnPause");
const btnStep = $("#btnStep");
const btnReset = $("#btnReset");

function formatInt(n) { return n.toLocaleString(undefined, { maximumFractionDigits: 0 }); }
function formatFloat(x, d=2) { return x.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }); }
function speedFromLogSlider(v) { return Math.floor(Math.pow(10, parseFloat(v))); }

function setSpeedLabel() { const s = speedFromLogSlider(speed.value); speedVal.textContent = formatInt(s); return s; }
function setNSitesLabel() { nSitesVal.textContent = formatInt(parseInt(nSites.value, 10)); }
function setMBlocksLabel() { mBlocksVal.textContent = formatInt(parseInt(mBlocks.value, 10)); }
function setHeightScaleLabel() { heightScaleVal.textContent = formatFloat(parseFloat(heightScale.value), 2); }
function setBetaLabel() { betaVal.textContent = betaInf.checked ? "∞" : formatFloat(parseFloat(beta.value), 2); }
function setRangeRLabel() { rangeRVal.textContent = formatInt(parseInt(rangeR.value, 10)); }

speed.addEventListener("input", () => setSpeedLabel());
nSites.addEventListener("input", () => setNSitesLabel());
mBlocks.addEventListener("input", () => setMBlocksLabel());
heightScale.addEventListener("input", () => setHeightScaleLabel());
beta.addEventListener("input", () => setBetaLabel());
betaInf.addEventListener("change", () => setBetaLabel());
rangeR.addEventListener("input", () => setRangeRLabel());

setSpeedLabel();
setNSitesLabel();
setMBlocksLabel();
setHeightScaleLabel();
setBetaLabel();
setRangeRLabel();

// Simulation state
let rng = new RNG(seedInput.value);
let model = new BetaBD(parseInt(nSites.value, 10), rng);
let events = new EventRing(parseInt(mBlocks.value, 10));

function rebuildModel() {
  rng = new RNG(seedInput.value);
  model = new BetaBD(parseInt(nSites.value, 10), rng);
  events = new EventRing(parseInt(mBlocks.value, 10));
}

btnReseed.addEventListener("click", () => rebuildModel());

btnReset.addEventListener("click", () => {
  model.reset();
  events.resize(parseInt(mBlocks.value, 10));
});

nSites.addEventListener("change", () => {
  model.setSize(parseInt(nSites.value, 10));
  model.reset();
  events.resize(parseInt(mBlocks.value, 10));
});

mBlocks.addEventListener("change", () => {
  events.resize(parseInt(mBlocks.value, 10));
});

let paused = false;
btnPause.addEventListener("click", () => {
  paused = !paused;
  btnPause.textContent = paused ? "Resume" : "Pause";
});
btnStep.addEventListener("click", () => {
  if (!paused) return;
  tickOnce(1/60, true);
});
document.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    paused = !paused;
    btnPause.textContent = paused ? "Resume" : "Pause";
  }
});

// FPS meter
let lastFrame = performance.now();
let fpsEMA = null;

// Event accumulator
let accumulator = 0;

function tickOnce(dtSeconds, forceStep=false) {
  const N = model.N;
  const blocksPerSec = speedFromLogSlider(speed.value);
  accumulator += blocksPerSec * dtSeconds;

  const maxEventsPerFrame = 250000;
  let nEvents = Math.min(maxEventsPerFrame, Math.floor(accumulator));
  if (forceStep) nEvents = Math.min(nEvents, 20000);
  accumulator -= nEvents;

  const betaValNum = parseFloat(beta.value);
  const betaInfOn = betaInf.checked;

  const useRates = usePaperRates.checked;
  const rNeighbour = 1;
  const rCenter = useRates ? 2 : 1;

  const exact = exactClocks.checked;
  const R = parseInt(rangeR.value, 10);
  const avgOn = avgMode.checked;

  if (!paused || forceStep) {
    for (const ev of model.batch(nEvents, betaValNum, betaInfOn, rNeighbour, rCenter, R, avgOn, exact)) {
      events.push(ev.x, ev.yNew, model.eventCount);
    }
  }

  const st = model.stats();
  hudEvents.textContent = formatInt(model.eventCount);
  hudSweeps.textContent = formatFloat(model.microTime, 2);
  hudMean.textContent = formatFloat(st.mean, 2);
  hudMinMax.textContent = `${formatInt(st.min)} / ${formatInt(st.max)}`;

  view.resizeToDisplaySize();
  view.clear();

  const doBrick = toggleBrick.checked;
  const doSurface = toggleSurface.checked;

  const center = subtractDrift.checked;
  const centerValue = st.mean;

  let sharedBounds = null;

  if (doBrick) {
    hudMode.textContent = doSurface ? "Brick + Surface" : "Brick";
    const eventsArray = Array.from(events.iter());
    sharedBounds = view.drawBricks(eventsArray, {
      N,
      autoScale: autoScale.checked,
      pxPerUnit: parseFloat(heightScale.value),
      center,
      centerValue,
    });
  } else {
    hudMode.textContent = "Surface";
  }

  if (doSurface) {
    view.drawSurface(model.h, {
      autoScale: autoScale.checked,
      pxPerUnit: parseFloat(heightScale.value),
      center,
      centerValue,
      yMin: sharedBounds ? sharedBounds.yMin : null,
      yMax: sharedBounds ? sharedBounds.yMax : null,
    });
  }
}

function animate(now) {
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;

  const fps = 1 / Math.max(1e-6, dt);
  fpsEMA = fpsEMA === null ? fps : (0.90 * fpsEMA + 0.10 * fps);
  hudFps.textContent = formatFloat(fpsEMA, 1);

  tickOnce(dt, false);
  requestAnimationFrame(animate);
}

requestAnimationFrame(animate);