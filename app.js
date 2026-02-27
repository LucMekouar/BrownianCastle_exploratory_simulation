// Brownian Castle / β-Ballistic Deposition simulator
// ---------------------------------------------------
// Range-R extension implemented as additional Poisson clocks:
// For each site x, there are 2R neighbour clocks (±1..±R) and a +1 clock.
//
// When a clock "rings", it proposes a candidate height y_i.
// We choose among candidates via softmax:
//   P(i) ∝ r_i * exp(β y_i)
// Then set h(x) ← y_i.
//
// Averaging checkbox:
//   - neighbour clock ±k proposes either h(x±k) (default) OR round((h(x)+h(x±k))/2) if averaging on
//   - +1 clock always proposes h(x)+1 (unchanged)

const $ = (sel) => /** @type {HTMLElement} */ (document.querySelector(sel));

const clamp01 = (t) => Math.max(0, Math.min(1, t));
const lerp = (a, b, t) => a + (b - a) * t;

// Piecewise linear gradient: green -> purple -> blue (castle palette).
function castleRGB(t) {
  t = clamp01(t);
  const c0 = [72, 170, 102];   // green
  const c1 = [176, 102, 204];  // purple
  const c2 = [96, 142, 230];   // blue
  let a, b, u;
  if (t < 0.55) { a = c0; b = c1; u = t / 0.55; }
  else { a = c1; b = c2; u = (t - 0.55) / 0.45; }
  return [
    Math.round(lerp(a[0], b[0], u)),
    Math.round(lerp(a[1], b[1], u)),
    Math.round(lerp(a[2], b[2], u)),
  ];
}

// --------- Small deterministic PRNG ----------
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
  next() { return this._rand(); }             // float in [0,1)
  int(n) { return (this._rand() * n) | 0; }   // integer 0..n-1
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

// --------- β-BD model on a periodic ring ----------
class BetaBD {
  /** @param {number} N @param {RNG} rng */
  constructor(N, rng) {
    this.setSize(N);
    this.rng = rng;
    this.eventCount = 0;
    this.microTime = 0; // sweeps or exact (Gillespie)
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
   * One event at a uniformly chosen site x.
   * Range-R extension: candidates are {+1} ∪ {±k clocks, k=1..R}.
   * @returns {{x:number, yNew:number}}
   */
  stepEvent(beta, betaInf, rNeighbor, rCenter, R, avgOn) {
    const N = this.N;
    const x = this.rng.int(N);
    const idx = (i) => (i + N) % N;

    const hx = this.h[x];

    // Build candidates and rates
    // Candidate 0 will be the center (+1) move.
    // Then we append 2R neighbour candidates: +1..+R, -1..-R (order irrelevant).
    const nCand = 1 + 2 * R;
    const y = new Float64Array(nCand);
    const r = new Float64Array(nCand);

    // Center clock (+1)
    y[0] = hx + 1;
    r[0] = rCenter;

    // Neighbour clocks
    let j = 1;
    for (let k = 1; k <= R; k++) {
      const hr = this.h[idx(x + k)];
      const hl = this.h[idx(x - k)];

      // default neighbour update: copy neighbour
      // averaging update: average(h(x), h(x±k)) (rounded later)
      y[j] = avgOn ? 0.5 * (hx + hr) : hr;
      r[j] = rNeighbor;
      j++;

      y[j] = avgOn ? 0.5 * (hx + hl) : hl;
      r[j] = rNeighbor;
      j++;
    }

    // Choose an index
    let choice = 0;

    if (betaInf) {
      // deterministic max with random tie-breaking
      let m = y[0];
      for (let i = 1; i < nCand; i++) if (y[i] > m) m = y[i];

      // collect argmax set
      const cands = [];
      for (let i = 0; i < nCand; i++) if (y[i] === m) cands.push(i);
      choice = cands[this.rng.int(cands.length)];
    } else {
      // softmax with numerical stabilisation (shift by max)
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

    // Apply chosen update
    // - Center move is integer already.
    // - Neighbour candidates may be half-integers if averaging is on: round to nearest integer.
    const yNew = (choice === 0) ? (hx + 1) : Math.round(y[choice]);
    this.h[x] = yNew;
    this.eventCount++;

    return { x, yNew };
  }

  /** Batch events generator */
  *batch(nEvents, beta, betaInf, rNeighbor, rCenter, R, avgOn, exactClocks) {
    const N = this.N;
    for (let i = 0; i < nEvents; i++) {
      const ev = this.stepEvent(beta, betaInf, rNeighbor, rCenter, R, avgOn);

      if (exactClocks) {
        // total event rate is N (one site chosen per event): Δt ~ Exp(N)
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

// --------- Canvas rendering ----------
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

  drawSurface(heights, opts) {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;

    let min = heights[0], max = heights[0];
    for (let i = 0; i < heights.length; i++) {
      const v = heights[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const mean = opts.center ? opts.centerValue : 0;

    let yMin = min - mean;
    let yMax = max - mean;
    if (yMax - yMin < 2) { yMax += 1; yMin -= 1; }
    const pad = 0.08 * (yMax - yMin);
    yMin -= pad; yMax += pad;

    const scaleY = opts.autoScale ? (H / (yMax - yMin)) : (opts.pxPerUnit * this.pixelRatio);
    const scaleX = W / heights.length;

    ctx.save();
    ctx.lineWidth = Math.max(1, 1.5 * this.pixelRatio);
    ctx.strokeStyle = "rgba(96,142,230,0.95)";
    ctx.beginPath();
    for (let i = 0; i < heights.length; i++) {
      const x = (i + 0.5) * scaleX;
      const yVal = (heights[i] - mean);
      const y = H - (yVal - yMin) * scaleY;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  drawBricks(eventsArray, opts) {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    if (!eventsArray || eventsArray.length === 0) return;

    // bounds from events (fast-ish)
    let yMin = eventsArray[0].y, yMax = eventsArray[0].y;
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

const usePaperRates = /** @type {HTMLInputElement} */ ($("#usePaperRates"));
const subtractDrift = /** @type {HTMLInputElement} */ ($("#subtractDrift"));
const seedInput = /** @type {HTMLInputElement} */ ($("#seed"));
const btnReseed = $("#btnReseed");
const exactClocks = /** @type {HTMLInputElement} */ ($("#exactClocks"));

const rangeR = /** @type {HTMLInputElement} */ ($("#rangeR"));
const rangeRVal = $("#rangeRVal");
const avgMode = /** @type {HTMLInputElement} */ ($("#avgMode"));

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

speed.addEventListener("input", () => setSpeedLabel());
nSites.addEventListener("input", () => setNSitesLabel());
mBlocks.addEventListener("input", () => setMBlocksLabel());
heightScale.addEventListener("input", () => setHeightScaleLabel());
beta.addEventListener("input", () => setBetaLabel());
betaInf.addEventListener("change", () => setBetaLabel());

rangeR.addEventListener("input", () => { rangeRVal.textContent = rangeR.value; });
rangeRVal.textContent = rangeR.value;

setSpeedLabel();
setNSitesLabel();
setMBlocksLabel();
setHeightScaleLabel();
setBetaLabel();

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

// Main loop accumulator for events per second
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

  // Rates:
  // - paper normalisation uses rCenter=2, neighbour clocks each rate 1
  // - otherwise: all clocks rate 1 (still works)
  const useRates = usePaperRates.checked;
  const rNeighbor = 1;
  const rCenter = useRates ? 2 : 1;

  const exact = exactClocks.checked;

  const R = parseInt(rangeR.value, 10);
  const avgOn = avgMode.checked;

  if (!paused || forceStep) {
    for (const ev of model.batch(nEvents, betaValNum, betaInfOn, rNeighbor, rCenter, R, avgOn, exact)) {
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

  if (doBrick) {
    hudMode.textContent = doSurface ? "Brick + Surface" : "Brick";
    const eventsArray = Array.from(events.iter());
    view.drawBricks(eventsArray, {
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