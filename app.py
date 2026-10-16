import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

# ---------------------------------------------------------------
# Load the model once, when the server starts
# (put the .pkl file in the same folder as app.py)
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_FILES = ["medical.pkl", "midical.pkl"]

model = None
for file_name in MODEL_FILES:
    path = BASE_DIR / file_name
    if path.exists():
        with open(path, "rb") as f:  # only load pickle files you trust
            model = pickle.load(f)
        break

DEFAULT_FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
FEATURES = list(getattr(model, "feature_names_in_", DEFAULT_FEATURES))

LABELS = {
    "age": "Age",
    "sex": "Sex",
    "bmi": "BMI",
    "children": "Children",
    "smoker": "Smoking",
    "region": "Region",
}

# Allowed range for every input (min, max)
LIMITS = {
    "age": (18, 100),
    "sex": (0, 1),
    "bmi": (10, 60),
    "children": (0, 10),
    "smoker": (0, 1),
    "region": (0, 3),
}


# ---------------------------------------------------------------
# Page: HTML + CSS + JavaScript, all inside this file
# ---------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MediCost - Medical Charges Estimator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Figtree:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --ink: #14272e;
    --muted: #5d7279;
    --page: #f2f7f8;
    --card: #ffffff;
    --line: #dbe7ea;
    --teal: #0e7c86;
    --teal-dark: #0a5c64;
    --teal-soft: #e3f3f4;
    --coral: #e2674a;
    --coral-soft: #fdece7;
    --radius: 18px;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
    font-family: 'Figtree', system-ui, sans-serif;
    background: var(--page);
    color: var(--ink);
    line-height: 1.5;
}

/* ---------- Header ---------- */
.hero {
    background: var(--teal-dark);
    color: #fff;
    padding: 44px 24px 96px;
}
.hero-inner { max-width: 1100px; margin: 0 auto; }
.brand {
    display: flex; align-items: center; gap: 10px;
    font-weight: 700; letter-spacing: 0.02em; margin-bottom: 26px;
}
.brand-icon {
    width: 38px; height: 38px; border-radius: 11px;
    background: rgba(255,255,255,0.14);
    display: grid; place-items: center;
}
.hero h1 {
    font-family: 'Fraunces', serif;
    font-size: clamp(2rem, 5vw, 3.1rem);
    line-height: 1.1; margin: 0 0 12px; max-width: 640px;
}
.hero p { margin: 0; color: #bfe3e6; font-size: 1.05rem; max-width: 560px; }

/* ---------- Layout ---------- */
.wrap {
    max-width: 1100px; margin: -64px auto 0; padding: 0 24px 40px;
    display: grid; grid-template-columns: 1.15fr 1fr; gap: 24px;
    align-items: start;
}
.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 30px;
    box-shadow: 0 14px 34px rgba(14, 60, 70, 0.08);
}
.card h2 {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem; margin: 0 0 4px;
}
.sub { color: var(--muted); margin: 0 0 24px; font-size: 0.95rem; }

/* ---------- Form ---------- */
.field { margin-bottom: 22px; }
.field-top {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 8px;
}
.field .title { font-weight: 600; font-size: 0.95rem; }
.value-badge {
    font-weight: 700; color: var(--teal-dark);
    background: var(--teal-soft); padding: 2px 12px; border-radius: 999px;
    font-size: 0.9rem;
}
.hint { color: var(--muted); font-size: 0.82rem; margin-top: 6px; }

input[type=range] {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 8px; border-radius: 999px;
    background: var(--line); outline: none;
}
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 24px; height: 24px; border-radius: 50%;
    background: var(--teal); border: 4px solid #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25); cursor: pointer;
}
input[type=range]::-moz-range-thumb {
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--teal); border: 4px solid #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25); cursor: pointer;
}

.pills { display: flex; flex-wrap: wrap; gap: 8px; }
.pills input { position: absolute; opacity: 0; pointer-events: none; }
.pills label { cursor: pointer; }
.pills span {
    display: inline-block; padding: 10px 18px; border-radius: 12px;
    border: 1.5px solid var(--line); background: #fff;
    font-weight: 600; font-size: 0.93rem; color: var(--muted);
    transition: all 0.15s ease;
}
.pills label:hover span { border-color: var(--teal); color: var(--teal-dark); }
.pills input:checked + span {
    background: var(--teal); border-color: var(--teal); color: #fff;
}
.pills input:focus-visible + span { outline: 3px solid #9fd5da; }
.pills.danger input:checked + span.yes {
    background: var(--coral); border-color: var(--coral);
}

.two { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }

button.go {
    width: 100%; margin-top: 8px; padding: 16px;
    background: var(--teal); color: #fff; border: none;
    border-radius: 14px; font: 700 1.05rem 'Figtree', sans-serif;
    cursor: pointer; transition: background 0.15s, transform 0.15s;
}
button.go:hover { background: var(--teal-dark); transform: translateY(-1px); }
button.go:disabled { opacity: 0.7; cursor: wait; transform: none; }

/* ---------- Result panel ---------- */
.result { position: sticky; top: 24px; min-height: 420px; }
.empty { text-align: center; padding: 50px 10px; color: var(--muted); }
.empty svg { margin-bottom: 14px; }
.empty h3 { margin: 0 0 6px; color: var(--ink); font-size: 1.1rem; }

.amount-label {
    color: var(--muted); font-weight: 600; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.amount {
    font-family: 'Fraunces', serif;
    font-size: clamp(2.6rem, 6vw, 3.4rem);
    color: var(--teal-dark); line-height: 1.1; margin: 4px 0 4px;
}
.amount-note { color: var(--muted); font-size: 0.9rem; margin-bottom: 22px; }

.chip {
    display: inline-block; padding: 5px 14px; border-radius: 999px;
    font-weight: 700; font-size: 0.82rem; margin-bottom: 18px;
}
.chip.ok { background: var(--teal-soft); color: var(--teal-dark); }
.chip.warn { background: var(--coral-soft); color: #a8412a; }

.break-title {
    font-weight: 700; font-size: 0.95rem; margin: 6px 0 12px;
    padding-top: 18px; border-top: 1px solid var(--line);
}
.row { margin-bottom: 12px; }
.row-top {
    display: flex; justify-content: space-between;
    font-size: 0.9rem; margin-bottom: 4px;
}
.row-top b { font-weight: 600; }
.bar-track { height: 8px; background: #edf3f4; border-radius: 999px; overflow: hidden; }
.bar { height: 100%; border-radius: 999px; transition: width 0.5s ease; }
.bar.up { background: var(--coral); }
.bar.down { background: var(--teal); }
.bar.base { background: #9bb3b8; }

.disclaimer {
    margin-top: 18px; font-size: 0.8rem; color: var(--muted);
    background: #f6fafb; border-radius: 10px; padding: 10px 12px;
}
.error {
    background: var(--coral-soft); color: #a8412a;
    padding: 12px 14px; border-radius: 12px; font-weight: 600; font-size: 0.92rem;
}
.hidden { display: none; }
.fade { animation: fade 0.4s ease; }
@keyframes fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

footer {
    text-align: center; color: var(--muted); font-size: 0.85rem;
    padding: 0 24px 40px; max-width: 700px; margin: 0 auto;
}

/* ---------- Mobile ---------- */
@media (max-width: 860px) {
    .wrap { grid-template-columns: 1fr; }
    .result { position: static; min-height: 0; }
    .two { grid-template-columns: 1fr; gap: 0; }
    .card { padding: 22px; }
    .hero { padding-bottom: 84px; }
}
</style>
</head>
<body>

<header class="hero">
  <div class="hero-inner">
    <div class="brand">
      <div class="brand-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>
      </div>
      MediCost
    </div>
    <h1>Estimate yearly medical charges in seconds</h1>
    <p>Enter a few personal details and the trained regression model predicts the expected medical cost.</p>
  </div>
</header>

<main class="wrap">

  <!-- ===== Form ===== -->
  <section class="card">
    <h2>Patient details</h2>
    <p class="sub">Move the sliders and pick the options that apply.</p>

    <form id="form" autocomplete="off">

      <div class="field">
        <div class="field-top">
          <label class="title" for="age">Age</label>
          <span class="value-badge" id="ageOut">30 years</span>
        </div>
        <input type="range" id="age" name="age" min="18" max="64" step="1" value="30">
      </div>

      <div class="field">
        <div class="field-top">
          <label class="title" for="bmi">Body mass index (BMI)</label>
          <span class="value-badge" id="bmiOut">26.0</span>
        </div>
        <input type="range" id="bmi" name="bmi" min="15" max="50" step="0.1" value="26">
        <div class="hint" id="bmiHint">Overweight range</div>
      </div>

      <div class="two">
        <div class="field">
          <div class="field-top"><span class="title">Sex</span></div>
          <div class="pills">
            <label><input type="radio" name="sex" value="0" checked><span>Female</span></label>
            <label><input type="radio" name="sex" value="1"><span>Male</span></label>
          </div>
        </div>

        <div class="field">
          <div class="field-top"><span class="title">Smoker</span></div>
          <div class="pills danger">
            <label><input type="radio" name="smoker" value="0" checked><span>No</span></label>
            <label><input type="radio" name="smoker" value="1"><span class="yes">Yes</span></label>
          </div>
        </div>
      </div>

      <div class="field">
        <div class="field-top"><span class="title">Number of children</span></div>
        <div class="pills">
          <label><input type="radio" name="children" value="0" checked><span>0</span></label>
          <label><input type="radio" name="children" value="1"><span>1</span></label>
          <label><input type="radio" name="children" value="2"><span>2</span></label>
          <label><input type="radio" name="children" value="3"><span>3</span></label>
          <label><input type="radio" name="children" value="4"><span>4</span></label>
          <label><input type="radio" name="children" value="5"><span>5</span></label>
        </div>
      </div>

      <div class="field">
        <div class="field-top"><span class="title">Region</span></div>
        <div class="pills">
          <label><input type="radio" name="region" value="0" checked><span>Northeast</span></label>
          <label><input type="radio" name="region" value="1"><span>Northwest</span></label>
          <label><input type="radio" name="region" value="2"><span>Southeast</span></label>
          <label><input type="radio" name="region" value="3"><span>Southwest</span></label>
        </div>
      </div>

      <button type="submit" class="go" id="btn">Estimate charges</button>
    </form>
  </section>

  <!-- ===== Result ===== -->
  <section class="card result" id="result">

    <div class="empty" id="empty">
      <svg width="84" height="84" viewBox="0 0 24 24" fill="none" stroke="#0e7c86" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>
      <h3>Your estimate will appear here</h3>
      <div>Fill in the details and press <b>Estimate charges</b>.</div>
    </div>

    <div class="hidden fade" id="out">
      <div class="amount-label">Estimated yearly charges</div>
      <div class="amount" id="amount">$0</div>
      <div class="amount-note">Based on the values you entered</div>
      <div class="chip" id="chip"></div>

      <div class="break-title">What drives this estimate</div>
      <div id="rows"></div>

      <div class="disclaimer">
        This is a statistical estimate from a simple model. It is not medical, insurance or financial advice.
      </div>
    </div>

    <div class="error hidden" id="err"></div>
  </section>

</main>

<footer>
  Built with Flask and scikit-learn. Category values (sex, smoker, region) are numeric codes
  and must match the encoding used when the model was trained.
</footer>

<script>
const $ = (id) => document.getElementById(id);

// ----- live values for the sliders -----
function bmiText(v) {
    if (v < 18.5) return "Underweight range";
    if (v < 25)   return "Healthy weight range";
    if (v < 30)   return "Overweight range";
    return "Obesity range";
}
function updateSliders() {
    $("ageOut").textContent = $("age").value + " years";
    const b = parseFloat($("bmi").value);
    $("bmiOut").textContent = b.toFixed(1);
    $("bmiHint").textContent = bmiText(b);
}
$("age").addEventListener("input", updateSliders);
$("bmi").addEventListener("input", updateSliders);
updateSliders();

// ----- helpers -----
function money(n) {
    return "$" + Math.round(n).toLocaleString("en-US");
}
function showError(msg) {
    $("empty").classList.add("hidden");
    $("out").classList.add("hidden");
    $("err").textContent = msg;
    $("err").classList.remove("hidden");
}

// ----- send the form to the server -----
$("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("btn");
    btn.disabled = true;
    btn.textContent = "Calculating...";

    const data = {};
    new FormData($("form")).forEach((value, key) => { data[key] = value; });

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        const json = await res.json();

        if (!res.ok) {
            showError(json.error || "Something went wrong.");
        } else {
            render(json);
        }
    } catch (err) {
        showError("Could not reach the server. Please try again.");
    }

    btn.disabled = false;
    btn.textContent = "Estimate charges";
});

function render(r) {
    $("err").classList.add("hidden");
    $("empty").classList.add("hidden");
    $("out").classList.remove("hidden");

    $("amount").textContent = money(r.prediction);

    const chip = $("chip");
    chip.textContent = r.level;
    chip.className = "chip " + (r.level_type === "warn" ? "warn" : "ok");

    // bars: scale everything against the biggest value
    let biggest = 1;
    r.parts.forEach((p) => { biggest = Math.max(biggest, Math.abs(p.amount)); });

    let html = "";
    r.parts.forEach((p) => {
        const width = Math.max(2, (Math.abs(p.amount) / biggest) * 100);
        let cls = "base";
        let sign = "";
        if (p.kind === "feature") {
            cls = p.amount >= 0 ? "up" : "down";
            sign = p.amount >= 0 ? "+" : "-";
        } else if (p.amount < 0) {
            sign = "-";
        }
        html += '<div class="row">' +
                '<div class="row-top"><b>' + p.label + '</b><span>' + sign + money(Math.abs(p.amount)) + '</span></div>' +
                '<div class="bar-track"><div class="bar ' + cls + '" style="width:0%" data-w="' + width + '"></div></div>' +
                '</div>';
    });
    $("rows").innerHTML = html;

    // animate the bars
    setTimeout(() => {
        document.querySelectorAll(".bar").forEach((b) => { b.style.width = b.dataset.w + "%"; });
    }, 30);

    if (window.innerWidth < 860) {
        $("result").scrollIntoView({ behavior: "smooth", block: "start" });
    }
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def home():
    return Response(PAGE, mimetype="text/html")


@app.route("/health")
def health():
    # Render can use this to check that the app is alive
    return jsonify(status="ok", model_loaded=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error="Model file not found on the server."), 500

    data = request.get_json(silent=True) or {}

    # 1. read and validate every value
    values = {}
    for name in FEATURES:
        if name not in data:
            return jsonify(error="Missing value: " + name), 400
        try:
            number = float(data[name])
        except (TypeError, ValueError):
            return jsonify(error="Invalid number for " + name), 400

        low, high = LIMITS.get(name, (-1e9, 1e9))
        if number < low or number > high:
            message = LABELS.get(name, name) + " must be between " + str(low) + " and " + str(high)
            return jsonify(error=message), 400
        values[name] = number

    # 2. predict (keep the same column order as training)
    row = pd.DataFrame([[values[n] for n in FEATURES]], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    prediction = max(prediction, 0.0)

    # 3. split the estimate into parts (works because the model is linear)
    parts = []
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        parts.append({
            "label": "Base amount",
            "amount": float(model.intercept_),
            "kind": "base",
        })
        for i in range(len(FEATURES)):
            name = FEATURES[i]
            parts.append({
                "label": LABELS.get(name, name),
                "amount": float(model.coef_[i]) * values[name],
                "kind": "feature",
            })

    # 4. a simple label for the result
    if prediction < 8000:
        level, level_type = "Lower than average cost", "ok"
    elif prediction < 16000:
        level, level_type = "Around average cost", "ok"
    else:
        level, level_type = "Higher than average cost", "warn"

    return jsonify(
        prediction=prediction,
        level=level,
        level_type=level_type,
        parts=parts,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
