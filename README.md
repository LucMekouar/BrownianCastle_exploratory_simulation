# Brownian Castle / β-Ballistic Deposition

Interactive, browser-based simulator of a 1D growth model from Cannizzaro–Hairer (“The Brownian Castle”), with a tunable inverse temperature parameter β and an extended “range-R” version.

This is a **static site** (HTML/CSS/JS only), so it’s easy to host on **GitHub Pages**.

---

## Description

We simulate a height function on a 1D periodic lattice:

\[
h:\mathbb{Z}\to\mathbb{Z}.
\]

At each update, we pick a site \(x\) and consider candidates coming from **Poisson clocks**:

- a **+1 clock** at \(x\), proposing \(h(x)+1\)
- **neighbour clocks** at offsets \(\pm 1, \pm 2, \ldots, \pm R\), proposing values copied from those neighbours
  (optionally replaced by an averaging rule)

We choose one candidate using a soft-max rule controlled by the **inverse temperature** \( \beta \).  
- \( \beta = 0 \) is the **0-BD regime** studied in the paper.
- Large \( \beta \) approaches a deterministic “pick the largest candidate” rule.

The visualisation is a “brick view” point cloud (the castle look), optionally with a surface profile overlay.

---

## Live demo (GitHub Pages)

Once Pages is enabled, your demo should be available at:

`https://<your-username>.github.io/<repo-name>/`

Example (for this repo name):  
`https://lucmekouar.github.io/BC_simulation/`

---

## Controls (what you can change)

- **Blocks/sec** — how fast events are executed (visual speed only)
- **Columns (N)** — system size (number of lattice sites)
- **Blocks shown (M)** — number of most recent updates shown in brick view
- **β (inverse temperature)** — randomness vs “pick the biggest”
- **Neighbour range (R)** — number of neighbour clocks per side (±1…±R)
- **Averaging toggle** — changes neighbour updates to average with the current height
- **Paper normalisation** — uses the paper’s relative clock rates (centre clock twice as frequent)
- **Exact continuous time (Gillespie)** — uses exponential waiting times instead of “events/N”
- **Seed** — reproducible randomness

---

## Files

- `index.html` — page layout + controls + math explanation
- `styles.css` — styling
- `app.js` — simulation + rendering logic

---

## Run locally

Because the JavaScript is loaded as a module, you should serve it (don’t double-click the HTML file).

```bash
python -m http.server 8000
# open http://localhost:8000
