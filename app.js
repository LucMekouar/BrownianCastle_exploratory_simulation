// Brownian Castle / β-Ballistic Deposition simulator
// ---------------------------------------------------
// Implements the microscopic update rule from Cannizzaro–Hairer “The Brownian Castle”
// for β=0 (0-BD) and extends it to the β-softmax family.
//
// Model (one event at x):
//   yL = h(x-1), yC = h(x)+1, yR = h(x+1)
//   P(choose i) ∝ r_i * exp(β * y_i)
//   then h(x) ← y_i
//
// For numerical stability we compute exp(β*(y_i - max(y))) (shifted softmax).

const $ = (sel) => /** @type {HTMLElement} */ (document.querySelector(sel));

// --------- Small deterministic PRNG (mulberry32) ----------
// We hash an arbitrary seed string → uint32 then use mulberry32.
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

// --------- Event buffer for “brick view” ----------
class EventRing {
  /** @param {number} capacity */
  constructor(capacity) {
    this.resize(capacity);
  }
  /** @param {number} capacity */
  resize(capacity) {
    this.capacity = Math.max(1, capacity | 0);
    this.x = new Uint32Array(this.capacity);
    this.y = new Int32Array(this.capacity);
    this.type = new Uint8Array(this.capacity); // 0=L,1=C,2=R
    this.idx = new Uint32Array(this.capacity); // event index (monotone, for aging)
    this.head = 0;
    this.size = 0;
  }
  /** @param {number} x @param {number} y @param {number} type @param {number} idx */
  push(x, y, type, idx) {
    const i = this.head;
    this.x[i] = x >>> 0;
    this.y[i] = y | 0;
    this.type[i] = type & 255;
    this.idx[i] = idx >>> 0;
    this.head = (this.head + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }
  // Iterate from newest → oldest
  *iter() {
    for (let k = 0; k < this.size; k++) {
      const i = (this.head - 1 - k + this.capacity) % this.capacity;
      yield { x: this.x[i], y: this.y[i], type: this.type[i], idx: this.idx[i] };
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
    this.microTime = 0; // Either sweeps (events/N) or exact time if Gillespie
  }

  /** @param {number} N */
  setSize(N) {
    this.N = Math.max(4, N | 0);
    this.h = new Int32Array(this.N); // start at 0
  }

  reset() {
    this.h.fill(0);
    this.eventCount = 0;
    this.microTime = 0;
  }

  /** One event. Returns {x, yNew, type} for brick rendering. */
  stepEvent(beta, betaInf, rL, rC, rR) {
    const N = this.N;
    const x = this.rng.int(N);
    const xm = (x - 1 + N) % N;
    const xp = (x + 1) % N;

    const yL = this.h[xm];
    const yC = this.h[x] + 1;
    const yR = this.h[xp];

    let choice = 1; // default C

    if (betaInf) {
      // Deterministic max rule (with random tie-breaking)
      const m = Math.max(yL, yC, yR);
      const cands = [];
      if (yL === m) cands.push(0);
      if (yC === m) cands.push(1);
      if (yR === m) cands.push(2);
      choice = cands[this.rng.int(cands.length)];
    } else {
      // Shifted softmax: exp(beta*(y - max)) keeps numbers in [0,1] when beta>0.
      const m = Math.max(yL, yC, yR);
      const w0 = rL * Math.exp(beta * (yL - m));
      const w1 = rC * Math.exp(beta * (yC - m));
      const w2 = rR * Math.exp(beta * (yR - m));
      const s = w0 + w1 + w2;

      // Draw u in [0,1)
      const u = this.rng.next() * s;
      if (u < w0) choice = 0;
      else if (u < w0 + w1) choice = 1;
      else choice = 2;
    }

    const yNew = (choice === 0) ? yL : (choice === 1) ? yC : yR;
    this.h[x] = yNew;
    this.eventCount++;

    return { x, yNew, type: choice };
  }

  /** Batch events. If exactClocks, advance microTime by Exp(N) each event. */
  *batch(nEvents, beta, betaInf, rL, rC, rR, exactClocks) {
    const N = this.N;
    for (let i = 0; i < nEvents; i++) {
      const ev = this.stepEvent(beta, betaInf, rL, rC, rR);
      if (exactClocks) {
        // Gillespie waiting time for total rate N: Δt ~ Exp(N)
        const u = Math.max(1e-12, this.rng.next());
        this.microTime += -Math.log(u) / N;
      } else {
        // “sweeps” time
        this.microTime = this.eventCount / N;
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
    ctx.fillStyle = "#060a12";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /** @param {Int32Array} heights */
  drawSurface(heights, opts) {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;

    // Compute min/max for scaling.
    let min = heights[0], max = heights[0];
    for (let i = 0; i < heights.length; i++) {
      const v = heights[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }

    // Optional centering.
    const mean = opts.center ? opts.centerValue : 0;

    // Expand bounds a bit for aesthetics.
    let yMin = min - mean;
    let yMax = max - mean;
    if (yMax - yMin < 2) { yMax += 1; yMin -= 1; }
    const pad = 0.08 * (yMax - yMin);
    yMin -= pad; yMax += pad;

    // Scale.
    const scaleY = opts.autoScale ? (H / (yMax - yMin)) : (opts.pxPerUnit * this.pixelRatio);
    const scaleX = W / heights.length;

    // Draw faint grid
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    const gridN = 6;
    for (let k = 1; k < gridN; k++) {
      const y = (k / gridN) * H;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }
    ctx.restore();

    // Polyline.
    ctx.save();
    ctx.lineWidth = Math.max(1, 1.5 * this.pixelRatio);
    ctx.strokeStyle = "rgba(110,168,254,0.95)";
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

    // Baseline at y=0 (after centering), if visible.
    const y0 = H - (0 - yMin) * scaleY;
    if (y0 >= 0 && y0 <= H) {
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.10)";
      ctx.beginPath();
      ctx.moveTo(0, y0);
      ctx.lineTo(W, y0);
      ctx.stroke();
      ctx.restore();
    }

    return { yMin, yMax, mean, scaleY, scaleX };
  }

  /** Draw last M events as “bricks” (points). */
  drawBricks(eventsIter, opts) {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;

    // Determine bounds from a sample of events (fast) OR use provided bounds.
    let yMin = opts.yMin, yMax = opts.yMax;
    if (yMin === null || yMax === null) {
      yMin = 0; yMax = 1;
      let first = true;
      let count = 0;
      for (const ev of eventsIter) {
        const y = ev.y - (opts.center ? opts.centerValue : 0);
        if (first) { yMin = y; yMax = y; first = false; }
        else { if (y < yMin) yMin = y; if (y > yMax) yMax = y; }
        count++;
        if (count > 2000) break; // sample cap
      }
      if (yMax - yMin < 2) { yMax += 1; yMin -= 1; }
      const pad = 0.12 * (yMax - yMin);
      yMin -= pad; yMax += pad;
    }

    const scaleY = opts.autoScale ? (H / (yMax - yMin)) : (opts.pxPerUnit * this.pixelRatio);
    const scaleX = W / opts.N;

    // Re-iterate (generator was consumed if we sampled). So we expect a cached array of events.
    const events = opts.eventsArray; // newest→oldest
    if (!events || events.length === 0) return { yMin, yMax };

    const newestIdx = events[0].idx;
    const oldestIdx = events[events.length - 1].idx;
    const span = Math.max(1, newestIdx - oldestIdx);

    // Draw bricks. Older bricks fade out.
    ctx.save();
    for (let k = 0; k < events.length; k++) {
      const ev = events[k];
      const age01 = (newestIdx - ev.idx) / span; // 0=newest, 1=oldest
      const alpha = 1.0 - age01;
      const x = (ev.x + 0.5) * scaleX;
      const yVal = ev.y - (opts.center ? opts.centerValue : 0);
      const y = H - (yVal - yMin) * scaleY;

      // Color by type (L,C,R)
      ctx.fillStyle =
        ev.type === 1 ? `rgba(255,255,255,${0.12 + 0.75 * alpha})` :
        ev.type === 0 ? `rgba(110,168,254,${0.10 + 0.75 * alpha})` :
                        `rgba(255,179,71,${0.10 + 0.75 * alpha})`;

      const s = Math.max(1, 2.0 * this.pixelRatio);
      ctx.fillRect(x - s/2, y - s/2, s, s);
    }
    ctx.restore();

    return { yMin, yMax };
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

const btnPause = $("#btnPause");
const btnStep = $("#btnStep");
const btnReset = $("#btnReset");

function formatInt(n) {
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}
function formatFloat(x, d=2) {
  return x.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}
function speedFromLogSlider(v) {
  // slider is log10 blocks/sec in [1,6] => 10 .. 1,000,000
  return Math.floor(Math.pow(10, parseFloat(v)));
}
function setSpeedLabel() {
  const s = speedFromLogSlider(speed.value);
  speedVal.textContent = formatInt(s);
  return s;
}
function setNSitesLabel() {
  nSitesVal.textContent = formatInt(parseInt(nSites.value, 10));
}
function setMBlocksLabel() {
  mBlocksVal.textContent = formatInt(parseInt(mBlocks.value, 10));
}
function setHeightScaleLabel() {
  heightScaleVal.textContent = formatFloat(parseFloat(heightScale.value), 2);
}
function setBetaLabel() {
  betaVal.textContent = betaInf.checked ? "∞" : formatFloat(parseFloat(beta.value), 2);
}

speed.addEventListener("input", () => setSpeedLabel());
nSites.addEventListener("input", () => setNSitesLabel());
mBlocks.addEventListener("input", () => setMBlocksLabel());
heightScale.addEventListener("input", () => setHeightScaleLabel());
beta.addEventListener("input", () => setBetaLabel());
betaInf.addEventListener("change", () => setBetaLabel());

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

btnReseed.addEventListener("click", () => {
  rebuildModel();
});

btnReset.addEventListener("click", () => {
  model.reset();
  events.resize(parseInt(mBlocks.value, 10));
});

nSites.addEventListener("change", () => {
  const N = parseInt(nSites.value, 10);
  model.setSize(N);
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

  // Prevent “spiral of death” if tab was inactive
  const maxEventsPerFrame = 250000;
  let nEvents = Math.min(maxEventsPerFrame, Math.floor(accumulator));
  if (forceStep) nEvents = Math.min(nEvents, 20000); // when stepping, keep it gentle
  accumulator -= nEvents;

  const betaValNum = parseFloat(beta.value);
  const betaInfOn = betaInf.checked;

  const useRates = usePaperRates.checked;
  const rL = useRates ? 1 : 1;
  const rC = useRates ? 2 : 1;
  const rR = useRates ? 1 : 1;

  const exact = exactClocks.checked;

  // Run batch and record events for brick view
  if (!paused || forceStep) {
    for (const ev of model.batch(nEvents, betaValNum, betaInfOn, rL, rC, rR, exact)) {
      events.push(ev.x, ev.yNew, ev.type, model.eventCount);
    }
  }

  // Update HUD stats
  const st = model.stats();
  hudEvents.textContent = formatInt(model.eventCount);
  hudSweeps.textContent = formatFloat(model.microTime, 2);
  hudMean.textContent = formatFloat(st.mean, 2);
  hudMinMax.textContent = `${formatInt(st.min)} / ${formatInt(st.max)}`;

  // Render
  view.resizeToDisplaySize();
  view.clear();

  const doBrick = toggleBrick.checked;
  const doSurface = toggleSurface.checked;

  const center = subtractDrift.checked;
  const centerValue = st.mean;

  if (doBrick) {
    hudMode.textContent = doSurface ? "Brick + Surface" : "Brick";
    // Cache events newest→oldest for one pass rendering
    const eventsArray = Array.from(events.iter());
    view.drawBricks(events.iter(), {
      N,
      autoScale: autoScale.checked,
      pxPerUnit: parseFloat(heightScale.value),
      center,
      centerValue,
      yMin: null,
      yMax: null,
      eventsArray,
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

  // FPS exponential moving average
  const fps = 1 / Math.max(1e-6, dt);
  fpsEMA = fpsEMA === null ? fps : (0.90 * fpsEMA + 0.10 * fps);
  hudFps.textContent = formatFloat(fpsEMA, 1);

  tickOnce(dt, false);
  requestAnimationFrame(animate);
}

requestAnimationFrame(animate);