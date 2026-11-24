/* 混合精度训练看板前端：原生 JS + Canvas，无外部依赖。 */

const COLORS = {
  fp32: "#8595b5", near: "#5aa2ff", stoch: "#b58cff",
  ok: "#43d190", warn: "#ffb454", bad: "#ff6b6b", purple: "#b58cff",
};

const state = { runs: [], current: null, compare: null, metrics: [], cmpMetrics: [] };

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 4) => (x === null || x === undefined) ? "–"
  : (Math.abs(x) >= 1e4 || (Math.abs(x) > 0 && Math.abs(x) < 1e-3)
      ? x.toExponential(2) : Number(x).toFixed(d));

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

/* ------------------------------------------------------------ Canvas 折线图 */

function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawChart(cv, series, opts = {}) {
  const { ctx, w, h } = setupCanvas(cv);
  const pad = { l: 46, r: 12, t: 10, b: 22 };
  const allY = series.flatMap(s => s.data.filter(v => v !== null && isFinite(v)));
  const allX = series.flatMap(s => s.data.map((_, i) => i));
  if (!allY.length) {
    ctx.fillStyle = "#8595b5"; ctx.font = "12px monospace";
    ctx.fillText("暂无数据", w / 2 - 30, h / 2);
    return;
  }
  let ymin = opts.ymin !== undefined ? opts.ymin : Math.min(...allY);
  let ymax = opts.ymax !== undefined ? opts.ymax : Math.max(...allY);
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const n = Math.max(...series.map(s => s.data.length));
  const X = i => pad.l + (n <= 1 ? 0 : (i / (n - 1)) * (w - pad.l - pad.r));
  const Y = v => pad.t + (1 - (v - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);

  // 网格 + y 轴刻度
  ctx.strokeStyle = "#2c3a5c"; ctx.fillStyle = "#8595b5";
  ctx.font = "10px monospace"; ctx.lineWidth = 1;
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = ymin + (ymax - ymin) * i / ticks;
    const yy = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    const label = opts.log ? v.toExponential(1) : (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : v.toFixed(2));
    ctx.fillText(label, 4, yy + 3);
  }
  // x 轴刻度
  for (let i = 0; i <= 4; i++) {
    const xx = X(Math.round((n - 1) * i / 4));
    ctx.fillText(String(Math.round((n - 1) * i / 4)), xx - 8, h - 6);
  }

  // 跳过步竖线
  if (opts.skipSteps) {
    ctx.strokeStyle = "rgba(255,107,107,0.35)";
    opts.skipSteps.forEach(s => {
      ctx.beginPath(); ctx.moveTo(X(s), pad.t); ctx.lineTo(X(s), h - pad.b); ctx.stroke();
    });
  }

  // 数据线
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.6; ctx.beginPath();
    let started = false;
    s.data.forEach((v, i) => {
      if (v === null || !isFinite(v)) { started = false; return; }
      const xx = X(i), yy = Y(v);
      if (!started) { ctx.moveTo(xx, yy); started = true; }
      else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
  });
}

/* ------------------------------------------------------------ 数据渲染 */

async function loadRuns() {
  const data = await getJSON("/api/runs");
  state.runs = data.runs;
  const cur = $("runSel").value || state.runs.find(r => r.name.includes("near"))?.name
    || state.runs[0]?.name;
  $("runSel").innerHTML = state.runs.map(r =>
    `<option value="${r.name}">${r.name} (${r.steps}步)</option>`).join("");
  $("cmpSel").innerHTML = '<option value="">（不对比）</option>' +
    state.runs.map(r => `<option value="${r.name}">${r.name}</option>`).join("");
  if (cur && state.runs.some(r => r.name === cur)) $("runSel").value = cur;
}

async function refresh() {
  const run = $("runSel").value;
  const cmp = $("cmpSel").value;
  if (!run) return;
  const [{ metrics }, cmpData] = await Promise.all([
    getJSON(`/api/metrics?run=${encodeURIComponent(run)}`),
    cmp ? getJSON(`/api/metrics?run=${encodeURIComponent(cmp)}`) : Promise.resolve(null),
  ]);
  state.metrics = metrics;
  state.cmpMetrics = cmpData ? cmpData.metrics : [];
  render();
}

function pct(x) { return x === null || x === undefined ? "–" : (x * 100).toFixed(2) + "%"; }

function render() {
  const rows = state.metrics;
  if (!rows.length) return;
  const latest = rows[rows.length - 1];
  const skipped = rows.filter(r => r.skipped);

  $("modePill").textContent = `${latest.mode} · 舍入=${latest.rounding}`
    + (latest.scaler_fixed ? " · scale 已固定" : " · 动态缩放");
  $("stepPill").textContent = `step ${latest.step} / 共 ${rows.length} 步`;

  // KPI
  $("kScale").textContent = fmt(latest.scale, 2);
  $("kScale").className = "v " + (latest.scaler_fixed ? "warn" : "ok");
  $("kScaleSub").textContent = `事件 ${latest.event} · good_steps=${latest.scaler?.good_steps ?? "–"}`;
  $("kSkip").textContent = skipped.length;
  $("kSkip").className = "v " + (skipped.length ? "bad" : "ok");
  $("kSkipSub").textContent = skipped.length
    ? "跳过步: " + skipped.map(r => r.step).join(", ") : "无溢出跳步";
  $("kOvf").textContent = pct(latest.grad_overflow_ratio);
  $("kOvf").className = "v " + (latest.grad_overflow_ratio > 0 ? "bad" : "ok");
  $("kGap").textContent = latest.param_gap === null ? "–" : latest.param_gap.toExponential(2);

  // 舍入对照表
  $("nearEat").textContent = pct(latest.near_eaten_frac);
  $("stochEat").textContent = pct(latest.stoch_eaten_frac);
  $("nearZero").textContent = pct(latest.near_zero_frac);
  $("stochZero").textContent = pct(latest.stoch_zero_frac);

  // 图表
  const lossSeries = [{
    name: latest.run || $("runSel").value, color: COLORS.near,
    data: rows.map(r => r.loss),
  }];
  const legend = [];
  legend.push(`<span><i style="background:${COLORS.near}"></i>${$("runSel").value} 损失</span>`);
  if (state.cmpMetrics.length) {
    lossSeries.push({
      name: $("cmpSel").value, color: COLORS.fp32,
      data: state.cmpMetrics.map(r => r.loss),
    });
    legend.push(`<span><i style="background:${COLORS.fp32}"></i>${$("cmpSel").value} 损失</span>`);
  }
  $("lossLegend").innerHTML = legend.join("");
  drawChart($("cLoss"), lossSeries);

  drawChart($("cScale"), [{
    color: COLORS.near, data: rows.map(r => r.scale),
  }], { log: true, skipSteps: skipped.map(r => r.step) });

  drawChart($("cUnder"), [
    { color: COLORS.bad, data: rows.map(r => r.underflow_unscaled_zero_frac) },
    { color: COLORS.ok, data: rows.map(r => r.underflow_scaled_zero_frac) },
  ], { ymin: 0, ymax: 1, skipSteps: skipped.map(r => r.step) });

  drawChart($("cRound"), [
    { color: COLORS.warn, data: rows.map(r => r.near_eaten_frac) },
    { color: COLORS.purple, data: rows.map(r => r.stoch_eaten_frac) },
  ], { ymin: 0, ymax: 1, skipSteps: skipped.map(r => r.step) });

  const accData = rows.map(r => r.eval ? r.eval.acc_master_fp32 : null);
  const accSeries = [{ color: COLORS.ok, data: accData }];
  if (state.cmpMetrics.length) {
    accSeries.push({ color: COLORS.fp32,
      data: state.cmpMetrics.map(r => r.eval ? r.eval.acc_master_fp32 : null) });
  }
  drawChart($("cAcc"), accSeries, { ymin: 0, ymax: 1.01 });

  drawChart($("cEat"), [{
    color: COLORS.warn, data: rows.map(r => r.eaten_update_frac),
  }], { ymin: 0, ymax: 1, skipSteps: skipped.map(r => r.step) });

  // 激活表
  if (latest.activations) {
    $("actTable").querySelector("tbody").innerHTML = latest.activations.map(a => `
      <tr><td>${a.layer}</td><td>${a.dtype}</td>
      <td>${fmt(a.min, 3)}</td><td>${fmt(a.max, 3)}</td><td>${fmt(a.absmax, 3)}</td>
      <td>${fmt(a.headroom_to_fp16_max, 1)}</td>
      <td style="color:${a.has_nonfinite ? "var(--bad)" : "var(--ok)"}">
        ${a.has_nonfinite ? "是 ⚠" : "否"}</td></tr>`).join("");
  }

  // 事件流
  const events = rows.filter(r => r.skipped || r.event !== "hold").slice(-25).reverse();
  $("evtTable").querySelector("tbody").innerHTML = events.map(r => {
    const tag = r.skipped ? '<span class="tag skip">overflow/跳过</span>'
      : r.event === "growth" ? '<span class="tag grow">scale 翻倍</span>'
      : '<span class="tag hold">hold</span>';
    return `<tr><td>${r.step}</td><td>${tag}</td><td>${fmt(r.scale, 2)}</td>
      <td>${fmt(r.loss, 4)}</td><td>${r.injected_overflow ? "⚠注入" : ""}</td>
      <td>${r.skipped ? (r.params_unchanged ? "是" : "否!!") : "–"}</td></tr>`;
  }).join("");

  // 控制按钮状态
  $("fixBtn").classList.toggle("on", !!latest.scaler_fixed);
  $("freeBtn").classList.toggle("on", !latest.scaler_fixed);
}

/* ------------------------------------------------------------ 控制操作 */

async function sendControl(payload) {
  const run = $("runSel").value;
  await postJSON(`/api/control?run=${encodeURIComponent(run)}`, payload);
  $("ctrlNote").textContent = "已下发: " + JSON.stringify(payload) + "，下一个训练步生效。";
  await refresh();
}

$("fixBtn").onclick = () => sendControl({ fixed: true, scale: null });
$("freeBtn").onclick = () => sendControl({ fixed: false, scale: null });
$("setScaleBtn").onclick = () => {
  const v = parseFloat($("scaleInput").value);
  if (!(v > 0)) { alert("请输入正数 scale"); return; }
  sendControl({ fixed: true, scale: v });
};
$("refreshBtn").onclick = refresh;
$("runSel").onchange = refresh;
$("cmpSel").onchange = refresh;

let timer = null;
function armTimer() {
  clearInterval(timer);
  if ($("autoRefresh").checked) timer = setInterval(refresh, 1000);
}
$("autoRefresh").onchange = armTimer;

(async function init() {
  await loadRuns();
  // 默认对比项选 full_fp32
  if (state.runs.some(r => r.name === "full_fp32")) $("cmpSel").value = "full_fp32";
  await refresh();
  armTimer();
  window.addEventListener("resize", () => render());
})();
