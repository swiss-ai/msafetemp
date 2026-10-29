(function () {
"use strict";

/* Set this to the arXiv URL once the paper is public; the hero button switches on by itself. */
var PAPER_URL = "";

var D = JSON.parse(document.getElementById("site-data").textContent);
var ROOT = document.documentElement;
var RM = window.matchMedia("(prefers-reduced-motion: reduce)");
var SVGNS = "http://www.w3.org/2000/svg";

/* #top is the sticky header, which is always "in view", so the browser never scrolls to it */
document.addEventListener("click", function (e) {
  var a = e.target.closest && e.target.closest('a[href="#top"]');
  if (!a) return;
  e.preventDefault();
  window.scrollTo({ top: 0, behavior: RM.matches ? "auto" : "smooth" });
});

/* ---------------- helpers ---------------- */
function $(s, r) { return (r || document).querySelector(s); }
function el(tag, attrs, kids) {
  var n = document.createElement(tag);
  if (attrs) for (var k in attrs) {
    var val = attrs[k];
    if (val == null || val === false) continue;
    if (k === "class") n.className = val;
    else if (k === "text") n.textContent = val;
    else if (k.slice(0, 2) === "on") n.addEventListener(k.slice(2), val);
    else if (k === "data-src") continue;
    else n.setAttribute(k, val === true ? "" : val);
  }
  add(n, kids);
  return n;
}
function add(n, kids) {
  if (kids == null) return n;
  if (!Array.isArray(kids)) kids = [kids];
  kids.forEach(function (c) {
    if (c == null || c === false) return;
    n.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
  });
  return n;
}
function svg(tag, attrs, kids) {
  var n = document.createElementNS(SVGNS, tag), css = "";
  /* colors are CSS variables: fill/stroke go into style so the theme switch needs no redraw */
  if (attrs) for (var k in attrs) {
    if (attrs[k] == null) continue;
    if ((k === "fill" || k === "stroke") && String(attrs[k]).indexOf("var(") === 0) css += k + ":" + attrs[k] + ";";
    else n.setAttribute(k, attrs[k]);
  }
  if (css) n.setAttribute("style", css + (n.getAttribute("style") || ""));
  if (kids != null) (Array.isArray(kids) ? kids : [kids]).forEach(function (c) {
    if (c == null) return;
    n.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
  });
  return n;
}
function C(name) { return "var(--" + name + ")"; }
function fmt(n) { return n == null ? "" : Number(n).toLocaleString("en-US"); }
function pct(x) { return typeof x === "number" ? x.toFixed(1) : String(x); }
function num(x) { return typeof x === "number" ? x : (String(x).indexOf("<") === 0 ? 0.05 : parseFloat(x)); }
function lang(k) { return (D.langs && D.langs[k]) || { iso: "", name: k, dir: "ltr" }; }
function langName(k) { return lang(k).name || k; }
function iso(k) { var i = lang(k).iso || ""; return i.split("-")[0].toLowerCase(); }
function setLang(node, k) { var L = lang(k); if (L.iso) node.setAttribute("lang", L.iso); node.setAttribute("dir", L.dir || "ltr"); return node; }
function segment(container, items, current, onPick) {
  container.textContent = "";
  items.forEach(function (it) {
    container.appendChild(el("button", {
      type: "button", "aria-pressed": String(it.id === current), disabled: it.disabled || null, title: it.title || null,
      onclick: function () { if (!it.disabled) onPick(it.id); }
    }, it.label));
  });
}
function armColor(arm) { return { tr: "tr", p0: "tr", p1: "tr", or: "or", ph: "ph", jb: "jb", seed: "en", en: "en" }[arm] || "ink"; }

/* Template text with <slot> markers -> carrier spans + slot chips */
var SLOT_RE = /<([A-Za-z_][A-Za-z0-9_]*)>|⟨([A-Za-z_][A-Za-z0-9_]*)⟩/g;
function templateFrag(text, opt) {
  opt = opt || {};
  var f = document.createDocumentFragment(), last = 0, m;
  SLOT_RE.lastIndex = 0;
  while ((m = SLOT_RE.exec(text))) {
    if (m.index > last) f.appendChild(el("span", { class: "carrier" }, text.slice(last, m.index)));
    var name = m[1] || m[2];
    var label = "⟨" + name + "⟩";
    if (opt.onSlot) {
      f.appendChild(el("button", {
        type: "button", class: "slot" + (opt.pick === name ? " is-picked" : ""), "aria-pressed": String(opt.pick === name),
        title: "Show the values this slot takes", onclick: (function (nm) { return function () { opt.onSlot(nm); }; })(name)
      }, label));
    } else {
      f.appendChild(el("span", { class: "slot" + (opt.pick === name ? " is-picked" : "") }, label));
    }
    last = SLOT_RE.lastIndex;
  }
  if (last < text.length) f.appendChild(el("span", { class: "carrier" }, text.slice(last)));
  return f;
}
/* Filled text with [{start,end,slot,gloss}] spans -> text + highlighted fills */
function filledFrag(text, fills, arm) {
  var f = document.createDocumentFragment(), pos = 0;
  var list = (fills || []).filter(function (x) { return x && x.end > x.start; }).slice().sort(function (a, b) { return a.start - b.start; });
  list.forEach(function (x) {
    if (x.start < pos) return;
    if (x.start > pos) f.appendChild(document.createTextNode(text.slice(pos, x.start)));
    var slot = x.slot || x.name || null;
    var m = el("mark", { class: "fill", "data-arm": arm, "data-slot": slot }, text.slice(x.start, x.end));
    f.appendChild(m);
    var gl = x.gloss || x.gloss_value;
    if (slot) f.appendChild(el("span", { class: "sr" }, " (slot " + slot + (gl ? ", gloss: " + gl : "") + ")"));
    pos = x.end;
  });
  if (pos < text.length) f.appendChild(document.createTextNode(text.slice(pos)));
  return f;
}
function swapAnim(node) {
  if (RM.matches) return;
  node.classList.remove("swap"); void node.offsetWidth; node.classList.add("swap");
}
function paneLabel(text, arm, extra) {
  return el("div", { class: "pane-label" }, [
    el("span", { class: "label", style: arm && armColor(arm) !== "en" ? "color:var(--" + armColor(arm) + ")" : null }, text), extra || null
  ]);
}
function glossLine(text, isTemplate) {
  if (!text) return null;
  var p = el("p", { class: "gloss", lang: "en" });
  if (isTemplate) p.appendChild(templateFrag(text)); else p.appendChild(document.createTextNode(text));
  return p;
}
/* "⟨slot⟩ gloss · ⟨slot⟩ gloss" for the fills of one row (stage spans carry gloss_value, release rows carry gloss) */
function fillGlossText(spans) {
  var seen = {}, out = [];
  (spans || []).forEach(function (s) {
    var g = s.gloss_value || s.gloss, k = (s.slot || "") + "|" + g;
    if (g && !seen[k]) { seen[k] = 1; out.push((s.slot ? "⟨" + s.slot + "⟩ " : "") + g); }
  });
  return out.join("; ");
}
function fillGlosses(spans) { var t = fillGlossText(spans); return t ? el("p", { class: "gloss", lang: "en" }, t) : null; }
/* scroll a long text box so the line above its first mark is the top line, never a half-cut one */
function scrollToMark(p, m) {
  if (!p || !m) return;
  var lh = parseFloat(getComputedStyle(p).lineHeight) || 24;
  p.scrollTop = Math.max(0, (Math.floor((m.offsetTop - p.offsetTop) / lh) - 1) * lh);
}
function empty(title, body) { return el("div", { class: "empty" }, [el("b", null, title), body || null]); }

/* ---------------- theme ---------------- */
var THEME_KEY = "msafetemp-theme";
/* sun, moon, half-filled circle for "follow the system" */
var HALF = "M12 3a9 9 0 0 1 0 18z";
var THEME_ICON = {
  light: ["c4", "M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4L6 18M18 6l1.4-1.4"],
  dark: ["M20.5 13.5A8.5 8.5 0 1 1 10.5 3.5a6.6 6.6 0 0 0 10 10z"],
  system: ["c9", HALF]
};
var themeBtn = $("#theme-btn");
function storedTheme() { try { var t = localStorage.getItem(THEME_KEY); return t === "light" || t === "dark" ? t : "system"; } catch (e) { return "system"; } }
function applyTheme(t) {
  if (t === "system") ROOT.removeAttribute("data-theme"); else ROOT.setAttribute("data-theme", t);
  try { if (t === "system") localStorage.removeItem(THEME_KEY); else localStorage.setItem(THEME_KEY, t); } catch (e) {}
  var label = { system: "System", light: "Light", dark: "Dark" }[t];
  var mode = t === "system" ? (window.matchMedia("(prefers-color-scheme: dark)").matches ? " (dark)" : " (light)") : "";
  var s = svg("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round", "aria-hidden": "true" }, THEME_ICON[t].map(function (d) {
    return d.charAt(0) === "c" ? svg("circle", { cx: 12, cy: 12, r: d.slice(1) }) : svg("path", { d: d, fill: d === HALF ? "currentColor" : null, stroke: d === HALF ? "none" : null });
  }));
  themeBtn.replaceChildren(s);
  label += mode;
  themeBtn.setAttribute("aria-label", "Color theme: " + label + ". Click to change.");
  themeBtn.title = "Theme: " + label;
}
var curTheme = storedTheme();
applyTheme(curTheme);
themeBtn.addEventListener("click", function () {
  var order = ["system", "light", "dark"];
  curTheme = order[(order.indexOf(curTheme) + 1) % 3];
  applyTheme(curTheme);
});

/* ---------------- header ---------------- */
(function () {
  var head = $(".site-head"), hero = $(".hero-fig");
  function onScroll() { head.classList.toggle("is-stuck", window.scrollY > 8); }
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
  /* mark the nav link of the section in view */
  var links = Array.prototype.slice.call(document.querySelectorAll(".nav a"));
  var secs = links.map(function (a) { return document.getElementById(a.getAttribute("href").slice(1)); });
  function current() {
    var y = 120, cur = null;
    secs.forEach(function (s, k) { if (s && s.getBoundingClientRect().top <= y) cur = k; });
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) cur = secs.length - 1;
    links.forEach(function (a, k) { if (k === cur) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); });
  }
  window.addEventListener("scroll", current, { passive: true }); current();
  var toc = $(".toc");
  toc.addEventListener("click", function (e) { if (e.target.tagName === "A") toc.open = false; });
  document.addEventListener("click", function (e) { if (toc.open && !toc.contains(e.target)) toc.open = false; });
  void hero;
})();

/* paper link */
(function () {
  var a = $("#paper-link");
  if (PAPER_URL) { a.href = PAPER_URL; $(".lt", a).textContent = "Paper"; a.removeAttribute("aria-disabled"); a.removeAttribute("role"); a.target = "_blank"; a.rel = "noopener"; }
  else a.setAttribute("title", "The arXiv link will be added when the paper is public");
})();

/* ---------------- hero ---------------- */
(function () {
  var items = D.hero || [];
  if (!items.length) return;
  var i = 0, timer = null, paused = false, hover = false;
  var fig = $("#hero-fig"), line = $("#hero-line"), lab = $("#hero-lang"), gl = $("#hero-gloss"), pauseBtn = $("#hero-pause");
  function show(k, anim) {
    i = (k + items.length) % items.length;
    var it = items[i];
    line.textContent = "";
    line.appendChild(templateFrag(it.text));
    setLang(line, it.lang);
    lab.textContent = "";
    lab.appendChild(el("b", null, iso(it.lang)));
    lab.appendChild(setLang(el("bdi", null, lang(it.lang).native || langName(it.lang)), it.lang));
    gl.textContent = "";
    gl.classList.toggle("is-note", !(it.gloss && it.lang !== "english"));
    if (it.gloss && it.lang !== "english") gl.appendChild(templateFrag(it.gloss));
    else gl.appendChild(document.createTextNode("Each slot replaces words that made the original prompt harmful. Only the text around the slots is translated; the slots are filled again inside each language."));
    if (anim) swapAnim(line);
  }
  function tick() { if (!paused && !hover && !document.hidden) show(i + 1, true); }
  function start() { stop(); if (!RM.matches && !paused) timer = setInterval(tick, 3200); }
  function stop() { if (timer) clearInterval(timer); timer = null; }
  $("#hero-prev").addEventListener("click", function () { show(i - 1, true); });
  $("#hero-next").addEventListener("click", function () { show(i + 1, true); });
  pauseBtn.addEventListener("click", function () {
    paused = !paused;
    pauseBtn.setAttribute("aria-pressed", String(paused));
    pauseBtn.setAttribute("aria-label", paused ? "Resume language cycling" : "Pause language cycling");
    pauseBtn.textContent = paused ? "Play" : "Pause";
    if (paused) stop(); else start();
  });
  fig.addEventListener("mouseenter", function () { hover = true; });
  fig.addEventListener("mouseleave", function () { hover = false; });
  fig.addEventListener("focusin", function () { hover = true; });
  fig.addEventListener("focusout", function () { hover = false; });
  if (RM.matches) { pauseBtn.hidden = true; }
  RM.addEventListener && RM.addEventListener("change", function () { pauseBtn.hidden = RM.matches; start(); });
  /* Reserve the tallest language's height so cycling never shifts the text below. */
  var body = $("#hero-body");
  function reserve() {
    var cur = i, mh = 0;
    body.style.minHeight = "";
    for (var k = 0; k < items.length; k++) { show(k, false); mh = Math.max(mh, body.offsetHeight); }
    body.style.minHeight = mh + "px";
    show(cur, false);
  }
  show(0, false);
  reserve();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(reserve);
  var rzT = null, lastW = window.innerWidth;
  window.addEventListener("resize", function () {
    if (window.innerWidth === lastW) return;
    lastW = window.innerWidth; clearTimeout(rzT); rzT = setTimeout(reserve, 150);
  });
  start();
})();

/* ---------------- method stepper ---------------- */
var Walk = (function () {
  var walks = D.walks || [];
  var P = D.pipeline || { stages: {} };
  var Y = P.yields || {};
  var STEPS = [
    { id: "seed", t: "English seed" },
    { id: "extract", t: "Extract slots", st: ["1"] },
    { id: "judge1", t: "Template", kind: "judge", st: ["2"] },
    { id: "translate", t: "Translate carrier", st: ["3", "5"] },
    { id: "judge2", t: "Translation", kind: "judge", st: ["4"] },
    { id: "templates", t: "Three templates", st: ["6", "7"] },
    { id: "branch", t: "Fill: TR, OR, PH", st: ["8", "10", "11", "13", "14"] },
    { id: "final", t: "Final variants", kind: "judge", st: ["9"] }
  ];
  var wi = Math.max(0, walks.findIndex(function (w) { return w.lang === "german"; }));
  var si = 6, timer = null;
  var seg = $("#walk-seg"), rail = $("#rail"), stage = $("#stage");

  function S(n) { return P.stages[n] || {}; }
  function longText(w) { return (w.stages[1] && w.stages[1].text.length > 300); }
  function srcName(w) {
    return /harmbench/i.test(w.track || "") ? "HarmBench" : "WildGuard";
  }
  function walkLabel(w) {
    var short = { german: "German", swiss_german: "Swiss German", ukrainian: "Ukrainian" }[w.lang] || langName(w.lang);
    return short + ", " + srcName(w);
  }
  /* one faint line under the tabs: where the selected example comes from */
  function drawSrc() {
    var w = walks[wi], n = $("#walk-src");
    if (!n) return;
    n.textContent = "";
  }
  function stageText(w, n, arm, isTemplate, langKey) {
    var s = w.stages[n];
    if (!s) return null;
    var p = el("div", { class: "pane-text ex" + (longText(w) ? " long" : "") });
    setLang(p, langKey);
    if (isTemplate) p.appendChild(templateFrag(s.text));
    else p.appendChild(filledFrag(s.text, s.spans, arm));
    return p;
  }
  function pane(label, arm, body, extra) {
    return el("div", { class: "pane" }, [paneLabel(label, arm, extra), body]);
  }
  function side(stNums, extra) {
    var box = el("div", { class: "stage-side" });
    (stNums || []).forEach(function (n) {
      var s = S(n); if (!s.stage) return;
      var dl = el("dl", null, [
        el("dt", null, "Stage"), el("dd", null, s.stage),
        el("dt", null, "Model"), el("dd", null, s.model)
      ]);
      box.appendChild(dl);
      var checks = s.checks || null;
      if (checks && checks.length) box.appendChild(el("ul", { class: "checks", "aria-label": "Judge criteria" }, checks.map(function (c) { return el("li", null, c); })));
      else if (s.code_criteria && n === "7") box.appendChild(el("ul", { class: "checks", "aria-label": "Judge criteria" }, s.code_criteria.map(function (c) { return el("li", null, c.toLowerCase().replace(/_/g, " ")); })));
    });
    if (extra) add(box, extra);
    return box;
  }
  function yieldP(text) { return el("p", { class: "faint", style: "font-size:13.5px" }, text.charAt(0).toUpperCase() + text.slice(1) + "."); }
  /* template gloss; a long one scrolls in its own box like the text above it */
  function tGloss(text, long) { var g = glossLine(text, true); if (g && long) g.classList.add("long"); return g; }

  function render() {
    var w = walks[wi]; if (!w) return;
    var step = STEPS[si];
    Array.prototype.forEach.call(rail.querySelectorAll("button"), function (b, k) {
      b.classList.toggle("done", k < si);
      if (k === si) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
    });
    $("#step-count").textContent = "Stage " + (si + 1) + " of " + STEPS.length + (timer ? "" : ". Replay runs from stage 1.");
    $("#step-prev").disabled = si === 0;
    $("#step-next").disabled = si === STEPS.length - 1;
    stage.textContent = "";
    var L = w.lang, isHB = /harmbench/i.test(w.track || "");
    var main = el("div", { class: "stage-main" });
    var title, what, state = el("div", { class: "state" }), sd = null;

    if (step.id === "seed") {
      title = "Start from a real harmful prompt";
      what = isHB ? "HarmBench contributes 200 standard behaviors, used only for evaluation. The underlined words carry the harm." :
        "WildGuardMix contributes 2,500 harmful prompts from its train split; this one is a training seed. The underlined words carry the harm.";
      state.appendChild(pane("English seed, " + srcName(w), "seed", stageText(w, 1, "seed", false, "english")));
      sd = side([], [el("dl", null, [el("dt", null, "Track"), el("dd", null, w.track), el("dt", null, "Harm"), el("dd", null, w.harm || ""), el("dt", null, "Seeds"), el("dd", { class: "num" }, isHB ? "200 behaviors" : "2,500 prompts")])]);
    } else if (step.id === "extract") {
      title = "Cut the harm out into typed slots";
      what = "The generator replaces the smallest harm-bearing spans with typed slots. Everything else is kept word for word, so the template itself is safe to translate.";
      state.appendChild(pane("English seed", "seed", stageText(w, 1, "seed", false, "english")));
      state.appendChild(pane("English template", null, stageText(w, 2, null, true, "english")));
      sd = side(["1"]);
    } else if (step.id === "judge1") {
      title = "Check the template before building on it";
      what = "A judge from another model family returns one yes or no per criterion; the decisions are combined in code.";
      state.appendChild(pane("English template", null, stageText(w, 2, null, true, "english")));
      sd = side(["2"]);
    } else if (step.id === "translate") {
      title = "Translate only the carrier text";
      what = "Slot names stay in English and stay in place, so the translator never sees the harmful entities. A patcher edits flagged spans for at most 3 rounds.";
      state.appendChild(pane("English template", null, stageText(w, 2, null, true, "english")));
      state.appendChild(pane("Native template, " + langName(L), null, stageText(w, 3, null, true, L)));
      sd = side(["3", "5"]);
    } else if (step.id === "judge2") {
      title = "Check the translation and the slots";
      what = "The second judge checks meaning, that every slot survived, and that the text is in the target language. English skips translation but runs both judges.";
      state.appendChild(pane("Native template, " + langName(L), null, stageText(w, 3, null, true, L)));
      sd = side(["4"], Y.valid_native_templates ? yieldP("valid native templates: WildGuard " + Y.valid_native_templates.wg + ", HarmBench " + Y.valid_native_templates.hb) : null);
    } else if (step.id === "templates") {
      title = "Paraphrase twice: three templates per seed and language";
      what = "Each paraphrase changes the wording while keeping every slot. All three are filled in the next stage.";
      var lg = longText(w);
      state.appendChild(pane("Main template", null, el("div", null, [stageText(w, 3, null, true, L), tGloss(w.stages[3] && w.stages[3].gloss, lg)])));
      if (w.paraphrases && w.paraphrases.length) {
        w.paraphrases.forEach(function (t, k) {
          var p = setLang(el("div", { class: "pane-text ex" + (lg ? " long" : "") }), L); p.appendChild(templateFrag(t));
          state.appendChild(pane("Paraphrase " + (k + 1), null, el("div", null, [p, tGloss(w.paraphraseGloss && w.paraphraseGloss[k], lg)])));
        });
      } else {
        state.appendChild(empty("Paraphrases not shown for this seed", "Only the main template is shown for this seed; the release holds two paraphrases for it."));
      }
      sd = side(["6", "7"]);
    } else if (step.id === "branch") {
      title = "Fill the slots three ways";
      what = "The same frame is filled with generic harmful values (TR), with benign values that keep the question form (OR), and with real local entities from the language's cultural card (PH).";
      main.appendChild(el("p", { class: "stage-title" }, title));
      main.appendChild(el("p", { class: "stage-what" }, what));
      main.appendChild(branchLanes(w));
      stage.appendChild(main);
      stage.style.gridTemplateColumns = "minmax(0,1fr)";
      var foot = el("div", { class: "stage-side", style: "grid-template-columns:repeat(auto-fit,minmax(220px,1fr));display:grid;gap:10px 28px" });
      if (Y.benign_candidates_accepted) foot.appendChild(yieldP("benign candidates accepted: WildGuard " + Y.benign_candidates_accepted.wg + ", HarmBench " + Y.benign_candidates_accepted.hb + "; " + (Y.direct_share_of_accepted_benign || "") + " of them DIRECT"));
      if (Y.templates_admitted_by_fit_judge) foot.appendChild(yieldP("templates admitted to PH by the fit judge: WildGuard " + Y.templates_admitted_by_fit_judge.wg + ", HarmBench " + Y.templates_admitted_by_fit_judge.hb));
      stage.appendChild(foot);
      scrollPanes();
      return;
    } else if (step.id === "final") {
      title = "Release only variants that pass all four checks";
      what = "TR and PH variants pass the same final judge. About 1% of generated harmful fills fail it and are dropped.";
      if (w.stages[4]) {
        var o4 = w.stages[4].outcome;
        state.appendChild(pane("TR, generic fill", "tr", stageText(w, 4, "tr", false, L), o4 ? el("span", { class: "verdict " + o4 }, "Olmo-3-7B-Instruct: " + o4) : null));
      }
      if (w.stages[5]) {
        var s5 = w.stages[5], o5 = s5.outcome;
        /* a long PH row gets its slot glosses and the note on what localization changed, not a full-length gloss */
        var ph5 = longText(w) ? [fillGlosses(s5.spans), s5.comment ? el("p", { class: "faint", style: "font-size:14.5px;margin-top:6px" }, s5.comment) : null]
          : [glossLine(s5.gloss || s5.comment)];
        state.appendChild(pane("PH, localized", "ph", el("div", null, [stageText(w, 5, "ph", false, L)].concat(ph5)), o5 ? el("span", { class: "verdict " + o5 }, "Olmo-3-7B-Instruct: " + o5) : null));
      }
      if (w.outcome) state.appendChild(el("p", { class: "stage-what", style: "margin-top:2px" }, [el("strong", null, "Outcome. "), w.outcome]));
      var yl = [];
      if (Y.harmful_fills_vs_ceiling) yl.push(yieldP("harmful fills reach " + Y.harmful_fills_vs_ceiling.wg + " (WildGuard) and " + Y.harmful_fills_vs_ceiling.hb + " (HarmBench) of the 15-per-language ceiling"));
      if (Y.localized_passing_final_judge) yl.push(yieldP("localized variants passing: " + Y.localized_passing_final_judge));
      sd = side(["9"], yl);
    }
    main.appendChild(el("p", { class: "stage-title" }, title));
    main.appendChild(el("p", { class: "stage-what" }, what));
    main.appendChild(state);
    stage.style.gridTemplateColumns = "";
    stage.appendChild(main);
    if (sd) stage.appendChild(sd);
    scrollPanes();
  }
  function scrollPanes() {
    Array.prototype.forEach.call(stage.querySelectorAll(".pane-text.long"), function (p) {
      var m = p.querySelector("mark, .slot");
      scrollToMark(p, m);
    });
  }
  function branchLanes(w) {
    var L = w.lang, wrap = el("div", { class: "lanes", style: "margin-top:16px" });
    var long = longText(w);
    function txt(n, arm) { var p = el("div", { class: "pane-text ex" + (long ? " long" : "") }); setLang(p, L); p.appendChild(filledFrag(w.stages[n].text, w.stages[n].spans, arm)); return p; }
    function slotsMeta(n) { return fillGlosses(w.stages[n] && w.stages[n].spans); }
    // TR
    var s8 = S("8");
    wrap.appendChild(el("div", { class: "lane", style: "--lane:var(--tr)" }, [
      el("h4", null, ["Harmful", el("span", { class: "tag" }, "TR")]),
      el("p", { class: "meta" }, "⟨slot⟩ ← generic harmful value. 5 fills × 3 templates, up to 15 per seed and language."),
      w.stages[4] ? txt(4, "tr") : empty("No generic fill shown"),
      slotsMeta(4)
    ]));
    // OR
    var s10 = S("10"), orLane = el("div", { class: "lane", style: "--lane:var(--or)" }, [
      el("h4", null, ["Benign", el("span", { class: "tag" }, "OR")]),
      el("p", { class: "meta" }, "⟨slot⟩ ← safe value, same question form. The judge keeps DIRECT and BOUNDED, rejects TOXIC.")
    ]);
    if (w.orPair) {
      var op = setLang(el("div", { class: "pane-text ex" + (long ? " long" : "") }), L); op.appendChild(filledFrag(w.orPair.text, w.orPair.fills, "or"));
      orLane.appendChild(op);
      orLane.appendChild(long ? fillGlosses(w.orPair.fills) : glossLine(w.orPair.gloss));
      /* say so only when the benign row was not generated on the TR row's own template */
      var sameTpl = !w.orTemplate || !w.stages[3] || w.orTemplate === w.stages[3].text;
      orLane.appendChild(el("p", { class: "meta" }, "From the release (" + (w.orPair.cls || "").toUpperCase() + ")" + (sameTpl ? ", on the same template as the TR row." :
        ". This row was generated on a second stored copy of the " + langName(L) + " template, so its wording differs slightly from the TR row.")));
    } else {
      orLane.appendChild(empty("No benign pair shown for this seed"));
    }
    wrap.appendChild(orLane);
    // PH
    var s14 = S("14"), s13 = S("13");
    wrap.appendChild(el("div", { class: "lane", style: "--lane:var(--ph)" }, [
      el("h4", null, ["Localized", el("span", { class: "tag" }, "PH")]),
      el("p", { class: "meta" }, "⟨slot⟩ ← local entity. A fit judge (" + (s13.model || "") + ") picks entities; the writer may rephrase around them."),
      w.stages[5] ? txt(5, "ph") : empty("No localized variant shown"),
      slotsMeta(5)
    ]));
    // cultural card, feeding PH
    var card = D.card || {};
    var cardLane = el("div", { class: "lane card", style: "--lane:var(--ph)" }, [
      el("h4", null, ["Cultural card", el("span", { class: "tag" }, iso(L))]),
      el("p", { class: "meta" }, "One card per language, 58 in all, written by Gemma 4 31B IT and checked by Gemini Flash 3.6.")
    ]);
    if (w.cardEntities && w.cardEntities.length) {
      /* the card entries this PH row draws on: the roles the card allows, and the slot the fit judge put it in here */
      cardLane.appendChild(el("p", { class: "label" }, langName(L) + " card, entries used above"));
      w.cardEntities.forEach(function (e) {
        cardLane.appendChild(el("div", { class: "card-ent" }, [
          el("span", { class: "label" }, (e.category || "").replace(/_/g, " ")),
          setLang(el("b", null, e.entity), L), e.gloss ? el("div", { class: "faint" }, e.gloss) : null,
          e.roles && e.roles.length ? el("div", { class: "faint", style: "font-size:13px" }, "usable as: " + e.roles.map(function (r) { return "⟨" + r + "⟩"; }).join(", ")) : null,
          e.slot ? el("div", { class: "faint", style: "font-size:13px" }, "fills here: ⟨" + e.slot + "⟩") : null
        ]));
      });
    } else if (L === card.lang && card.entities && card.entities.length) {
      var e0 = card.entities[0];
      var ent = function (e) {
        return el("div", { class: "card-ent" }, [
          el("span", { class: "label" }, (e.category || "").replace(/_/g, " ")),
          el("b", { lang: "de" }, e.entity), " ", el("span", { class: "faint" }, "(" + e.gloss_en + ")"),
          e.usable_as && e.usable_as[0] ? el("div", { class: "faint", style: "font-size:13px" }, "usable as: " + e.usable_as[0].replace(/<([A-Za-z_]+)>/g, "⟨$1⟩")) : null
        ]);
      };
      cardLane.appendChild(el("p", { class: "label" }, "German card, " + card.total + " entities"));
      cardLane.appendChild(ent(e0));
      if (card.entities.length > 1) {
        cardLane.appendChild(el("details", null, [el("summary", null, (card.entities.length - 1) + " more entities and the card's fields")].concat(card.entities.slice(1, 3).map(ent)).concat([
          el("p", { class: "meta", style: "margin-top:6px" }, "Each entity lists its local context, grammatical form, and the slot roles it may and may not fill.")
        ])));
      }
    } else {
      cardLane.appendChild(el("p", { class: "meta" }, "Categories: " + (card.categories || []).map(function (c) { return c.replace(/_/g, " "); }).join(", ") + "."));
    }
    wrap.appendChild(cardLane);
    return wrap;
  }
  function drawSeg() { segment(seg, walks.map(function (w, k) { return { id: k, label: walkLabel(w) }; }), wi, function (k) { wi = k; stopReplay(); drawSeg(); render(); }); drawSrc(); }
  function go(k) { si = Math.max(0, Math.min(STEPS.length - 1, k)); render(); }
  function stopReplay() { if (timer) { clearInterval(timer); timer = null; $("#step-replay").textContent = "Replay"; } }
  function replay() {
    if (timer) { stopReplay(); return; }
    go(0);
    if (RM.matches) return;
    $("#step-replay").textContent = "Stop";
    timer = setInterval(function () {
      if (si >= STEPS.length - 1) { stopReplay(); return; }
      go(si + 1);
    }, 1700);
  }
  function init() {
    if (!walks.length) { stage.appendChild(empty("Walkthrough data missing", "Rebuild the page with data/examples.json or results.json present.")); return; }
    drawSeg();
    STEPS.forEach(function (s, k) {
      rail.appendChild(el("li", null, el("button", { type: "button", "data-kind": s.kind || null, onclick: function () { stopReplay(); go(k); } }, [
        el("span", { class: "n" }, String(k + 1)), el("span", { class: "t" }, [s.t, s.kind === "judge" ? el("span", { class: "jt", title: "Output checked by a judge model" }, "\u2020") : null])
      ])));
    });
    $("#step-prev").addEventListener("click", function () { stopReplay(); go(si - 1); });
    $("#step-next").addEventListener("click", function () { stopReplay(); go(si + 1); });
    $("#step-replay").addEventListener("click", replay);
    render();
  }
  init();
  return { show: function (langKey) { var k = walks.findIndex(function (w) { return w.lang === langKey; }); if (k >= 0) { wi = k; stopReplay(); drawSeg(); go(6); } } };
})();

/* ---------------- explorer ---------------- */
(function () {
  var B = D.behaviors || [];
  var root = $("#ex");
  if (!B.length) {
    $("#ex-variants").appendChild(empty("Examples not built yet", "data/examples.json was missing when this page was built."));
    return;
  }
  var LANGS = ["english", "german", "ukrainian", "japanese", "hindi", "arabic", "swahili", "yoruba"].filter(function (k) { return B.some(function (b) { return b.byLang[k]; }); });
  var st = { b: 0, lang: "german", arm: "tr", tpl: "main", pick: null };
  var ARMS = [{ id: "tr", label: "TR harmful" }, { id: "or", label: "OR benign" }, { id: "ph", label: "PH localized" }];
  var TPLS = [{ id: "main", label: "Main" }, { id: "paraphrase_1", label: "Paraphrase 1" }, { id: "paraphrase_2", label: "Paraphrase 2" }];
  var TPL_NAME = { main: "Main template", paraphrase_1: "Paraphrase 1", paraphrase_2: "Paraphrase 2", main_v2: "Main template, as stored with the benign rows" };
  var ARMKEY = { main: "tr", paraphrase_1: "p0", paraphrase_2: "p1" };

  function cur() { return B[st.b]; }
  function data() { return cur().byLang[st.lang]; }
  function variants() {
    var d = data(); if (!d) return [];
    var key = st.arm === "tr" ? ARMKEY[st.tpl] : st.arm;
    return d.variants[key] || [];
  }
  function controls() {
    segment($("#ex-beh"), B.map(function (b, k) { return { id: k, label: b.title }; }), st.b, function (k) { st.b = k; st.pick = null; renderAll(); });
    var lc = $("#ex-lang"); lc.textContent = "";
    LANGS.forEach(function (k) {
      lc.appendChild(el("button", { type: "button", class: "tab", "aria-pressed": String(k === st.lang), onclick: function () { st.lang = k; renderAll(true); } }, [iso(k), el("small", null, langName(k))]));
    });
    segment($("#ex-arm"), ARMS.map(function (a) { return { id: a.id, label: st.lang === "english" && a.id === "tr" ? "EN harmful" : a.label }; }), st.arm, function (a) { st.arm = a; renderAll(); });
    var d = data();
    segment($("#ex-tpl"), TPLS.map(function (t) {
      return { id: t.id, label: t.label, disabled: !(d && d.templates[t.id]) || (st.arm !== "tr" && t.id !== "main"), title: st.arm !== "tr" ? "OR and PH rows are built from the main template" : null };
    }), st.arm === "tr" ? st.tpl : "main", function (t) { st.tpl = t; renderAll(); });
    var tn = $("#ex-tpl-note");
    if (!tn) { tn = el("span", { class: "seg-note", id: "ex-tpl-note" }); $("#ex-tpl").parentNode.appendChild(tn); }
    tn.textContent = st.arm === "tr" ? "" : (st.arm === "or" ? "Benign" : "Localized") + " rows are built from the main template only.";
  }
  function seedPane() {
    var b = cur(), p = $("#ex-seed"); p.textContent = "";
    p.appendChild(paneLabel("English seed, HarmBench", "seed"));
    var t = el("div", { class: "pane-text ex", lang: "en" });
    t.appendChild(filledFrag(b.seed, b.seedSpans, "seed"));
    p.appendChild(t);
    p.appendChild(el("p", { class: "note-line" }, "Underlined: the spans the extractor turned into slots. HarmBench category: " + (b.category || "").replace(/_/g, " ") + "."));
  }
  function templatePane(anim) {
    var d = data(), p = $("#ex-template"); p.textContent = "";
    if (!d) { p.appendChild(empty("No data for this language")); return; }
    var vs = variants();
    var tk = st.arm === "tr" ? st.tpl : ((vs[0] && vs[0].tk) || "main");
    var t = d.templates[tk] || d.templates.main;
    p.appendChild(paneLabel((st.lang === "english" ? "English template, " : "Native template, ") + (TPL_NAME[tk] || tk), null, el("span", { class: "faint", style: "font-size:13px" }, "click a slot")));
    if (!t) { p.appendChild(empty("Template missing")); return; }
    var box = setLang(el("div", { class: "pane-text ex" }), st.lang);
    box.appendChild(templateFrag(t.text, { pick: st.pick, onSlot: function (nm) { st.pick = nm; templatePane(false); vocab(); } }));
    p.appendChild(box);
    if (st.lang !== "english") p.appendChild(glossLine(t.gloss, true));
    if (anim) swapAnim(box);
  }
  function variantPane() {
    var d = data(), p = $("#ex-variants"); p.textContent = "";
    var armLabel = { tr: st.lang === "english" ? "Harmful fills (EN)" : "Harmful fills (TR)", or: "Benign minimal pairs (OR)", ph: "Localized fills (PH)" }[st.arm];
    var vs = variants();
    if (st.arm === "ph" && vs.length) p.appendChild(el("p", { class: "note", style: "margin:0 0 12px" }, "Localized rows name real groups, movements and institutions of the language's region, as the dataset does. Rows with slurs are either shown with the slur masked and named real people replaced by a placeholder, or left out."));
    var avail = d ? (d.available[st.arm === "tr" ? ARMKEY[st.tpl] : st.arm] || 0) : 0;
    p.appendChild(paneLabel(armLabel + ", " + langName(st.lang), st.arm, vs.length ? el("span", { class: "faint", style: "font-size:13px" }, vs.length + " shown" + (avail > vs.length ? " of " + avail + " in the release" : "")) : null));
    if (!vs.length) {
      var why;
      if (st.arm === "ph" && st.lang === "english") why = empty("English has no PH arm", "Localization swaps in entities from a language's cultural card; English is the source language and has no card.");
      else if (st.arm === "ph" && d && d.phExcluded && Object.keys(d.phExcluded).length) {
        var ex = Object.keys(d.phExcluded).map(function (k) { return d.phExcluded[k] + " " + k.replace(/_/g, " "); }).join(", ");
        why = empty("Localized rows exist but are not shown here", "This page leaves out PH rows with slurs or pejoratives the paper does not print, calls for violence against named real people, or named living people as targets (" + ex + ").");
      } else if (st.arm === "ph") why = empty("No localized variant for this behavior and language", "The fit judge admits only templates where a card entity fits the slot; about 27% of HarmBench templates pass.");
      else if (st.arm === "or") why = empty("No benign pair for this behavior and language", "The benign judge rejects candidates that stay unsafe or lose their meaning; for HarmBench it accepts 61% overall.");
      else why = empty("No rows for this combination");
      p.appendChild(why);
      return;
    }
    var ul = el("ol", { class: "variants" });
    vs.forEach(function (v, k) {
      var t = setLang(el("div", { class: "pane-text ex" }), st.lang);
      t.appendChild(filledFrag(v.text, v.fills, st.arm === "tr" ? ARMKEY[st.tpl] : st.arm));
      ul.appendChild(el("li", null, [
        el("span", { class: "vi" }, String(k + 1)),
        el("div", null, [t, st.lang !== "english" ? glossLine(v.gloss) : null, v.cls ? el("p", { class: "label", style: "margin-top:4px" }, "benign class: " + v.cls) : null])
      ]));
    });
    p.appendChild(ul);
    var note = D.langNotes && D.langNotes[st.lang];
    if (note) p.appendChild(el("p", { class: "faint", style: "font-size:13px;margin-top:8px" }, note));
  }
  /* A strict minimal pair: TR and OR rows built on the same stored template with the same slots. */
  function matchedPair(lk) {
    var d = cur().byLang[lk]; if (!d) return null;
    var trs = d.variants.tr || [], ors = d.variants.or || [];
    for (var i = 0; i < ors.length; i++) {
      var o = ors[i];
      var t = trs.filter(function (x) { return (x.tk || "main") === (o.tk || "main") && slotKey(x) === slotKey(o); })[0];
      if (t) return { tr: t, or: o };
    }
    return null;
  }
  function slotKey(v) { return (v.fills || []).map(function (f) { return f.slot; }).sort().join("|"); }
  function pair() {
    var box = $("#ex-pair"); box.textContent = "";
    var lk = st.lang, mp = matchedPair(lk);
    if (!mp) {
      var alt = LANGS.filter(function (k) { return k !== "english" && matchedPair(k); }).concat(matchedPair("english") ? ["english"] : []);
      if (alt.length) {
        lk = alt[0]; mp = matchedPair(lk);
      }
    }
    $("#pair-lang").textContent = (mp ? langName(lk) + ", " : "") + cur().title;
    if (!mp) { box.appendChild(empty("No strict pair for this behavior", "None of the eight languages here has a benign row on the same template as its harmful rows for this behavior.")); return; }
    var trv = mp.tr, orv = mp.or;
    function side(v, arm, label) {
      var t = setLang(el("div", { class: "pane-text ex" }), lk); t.appendChild(filledFrag(v.text, v.fills, arm));
      return el("div", { class: "pane" }, [paneLabel(label, arm), t, lk !== "english" ? glossLine(v.gloss) : null]);
    }
    box.appendChild(el("div", { class: "pair" }, [side(trv, "tr", "Harmful (TR)"), side(orv, "or", "Benign (OR" + (orv.cls ? ", " + orv.cls.toUpperCase() : "") + ")")]));
    var slots = [];
    trv.fills.forEach(function (f) { if (slots.indexOf(f.slot) < 0) slots.push(f.slot); });
    function val(v, sl) { return v.fills.filter(function (x) { return x.slot === sl; })[0]; }
    var tb = el("tbody");
    slots.forEach(function (sl) {
      function cell(f, arm, head) {
        if (!f) return el("td", { class: "faint", "data-h": head }, "not filled");
        return el("td", { "data-h": head }, [setLang(el("span", { class: "ex", style: "color:var(--" + arm + ")" }, f.value || ""), lk), lk !== "english" && f.gloss ? el("p", { class: "gloss", style: "margin-top:2px" }, f.gloss) : null]);
      }
      tb.appendChild(el("tr", null, [el("td", null, el("span", { class: "slot" }, "⟨" + sl + "⟩")), cell(val(trv, sl), "tr", "Harmful value"), cell(val(orv, sl), "or", "Benign value")]));
    });
    box.appendChild(el("table", { class: "slot-table" }, [
      el("thead", null, el("tr", null, [el("th", null, "Slot"), el("th", null, "Harmful value"), el("th", null, "Benign value")])), tb
    ]));
  }
  function vocab() {
    var b = cur(), box = $("#ex-vocab"); box.textContent = "";
    var slots = (b.vocab && b.vocab.slots || []).filter(function (s) { return s.langs >= 5; });
    if (!slots.length) { box.appendChild(empty("No slot statistics for this behavior")); return; }
    var pick = slots.filter(function (s) { return s.slot === st.pick; })[0] || slots[0];
    var left = el("div", null, [
      el("p", { style: "font-size:15px" }, [el("span", { class: "num", style: "font-weight:600", "data-src": "examples.json behaviors[].slot_vocabulary" }, fmt(pick.distinct)), " distinct values for ", el("span", { class: "slot" }, "⟨" + pick.slot + "⟩"), " in " + fmt(pick.occ) + " rows across " + pick.langs + " languages."]),
      el("div", { class: "chips", style: "margin-top:12px" }, slots.map(function (s) {
        return el("button", { type: "button", class: "slot" + (s === pick ? " is-picked" : ""), "aria-pressed": String(s === pick), onclick: function () { st.pick = s.slot; templatePane(false); vocab(); } }, "\u27e8" + s.slot + "\u27e9");
      })),
      b.vocab.recomb ? el("p", { class: "faint", style: "font-size:13px;margin-top:12px" }, "The release holds " + fmt(b.vocab.rows) + " TR main-template rows for this behavior. Swapping the observed values between rows of the same language would give " + fmt(b.vocab.recomb) + " distinct prompts, so the fills are not a small fixed list.") : null
    ]);
    var list = el("ul", { class: "vocab-list" }, (pick.examples || []).map(function (e) {
      return el("li", null, [setLang(el("span", { class: "v ex" }, e.value), e.lang), el("span", { class: "m" }, [el("code", null, iso(e.lang)), " " + langName(e.lang)]), e.gloss && e.lang !== "english" ? el("span", { class: "gloss", style: "margin-top:0;font-size:13px" }, e.gloss) : null]);
    }));
    box.appendChild(el("div", { class: "vocab" }, [left, list.childNodes.length ? list : empty("No example values listed")]));
  }
  function renderAll(anim) {
    var d = data();
    if (st.arm === "tr" && d && !d.templates[st.tpl]) st.tpl = "main";
    controls(); seedPane(); templatePane(anim); variantPane(); pair(); vocab();
  }
  renderAll(false);
  void root;
})();

/* ---------------- data matrix ---------------- */
(function () {
  var M = D.matrix, T = D.tracks;
  if (!M) return;
  // seed split bars, to scale
  var sb = $("#splitbar");
  var maxSeeds = T.wg.seeds;
  function row(name, sub, segs) {
    var track = el("div", { class: "sb-track" });
    segs.forEach(function (s) {
      var small = s.n / maxSeeds < 0.16;
      track.appendChild(el("div", { class: "sb-seg " + s.cls, style: "width:" + (100 * s.n / maxSeeds).toFixed(2) + "%", title: s.title || null }, small ? null : s.label));
      if (small) track.appendChild(el("span", { class: "sb-out" }, s.label));
    });
    return el("div", { class: "sb-row" }, [el("div", { class: "sb-name" }, [el("b", null, name), el("span", { class: "faint" }, sub)]), track]);
  }
  sb.appendChild(row("WildGuardMix", fmt(T.wg.seeds) + " seeds → " + T.wg.langs + " languages", [
    { cls: "train", n: T.wg.trainSeeds, label: fmt(T.wg.trainSeeds) + " train seeds, " + fmt(T.wg.trainRows) + " rows" },
    { cls: "test", n: T.wg.testSeeds, label: fmt(T.wg.testSeeds) + " test", title: fmt(T.wg.testSeeds) + " held-out seeds, " + fmt(T.wg.testRows) + " rows" }
  ]));
  sb.appendChild(row("HarmBench", T.hb.seeds + " behaviors → " + T.hb.langs + " languages", [
    { cls: "eval", n: T.hb.seeds, label: T.hb.seeds + " eval", title: fmt(T.hb.rows) + " rows, evaluation only" }
  ]));

  var COLS = [
    { id: "Orig", label: "Orig", sub: "English seed", arm: "en", def: "Orig" },
    { id: "EN", label: "EN", sub: "English template", arm: "en", def: "EN" },
    { id: "TR", label: "TR", sub: "translated, generic fills", arm: "tr", def: "TR" },
    { id: "PH", label: "PH", sub: "localized fills", arm: "ph", def: "PH" },
    { id: "ORD", label: "OR direct", sub: "benign, safe to answer", arm: "or", def: "OR-DIRECT" },
    { id: "ORB", label: "OR bounded", sub: "benign, answer with care", arm: "or", def: "OR-BOUNDED" },
    { id: "JB", label: "JB wrapped", sub: "inside a jailbreak wrapper", arm: "jb", def: "-JB" }
  ];
  var table = $("#matrix"), detail = $("#cell-detail"), sel = { r: "hb", c: "PH" };
  var thead = el("thead", null, el("tr", null, [el("th", null, "Split")].concat(COLS.map(function (c) {
    return el("th", null, [el("span", { class: "sq", style: "background:var(--" + c.arm + ")" }), c.label, el("small", null, c.sub)]);
  }))));
  var tbody = el("tbody");
  /* only the train row gets a row-level use label; test rows mix ASR, OR and unscored cells, so each cell shows its own */
  var ROWUSE = { train: "used for training" };
  M.rows.forEach(function (r) {
    var tr = el("tr", null, el("th", { scope: "row" }, [r.name, el("small", null, r.sub), el("small", { class: "num" }, fmt(r.total) + " rows"), ROWUSE[r.use] ? el("span", { class: "row-use" }, ROWUSE[r.use]) : null]));
    COLS.forEach(function (c) {
      var cell = M.cells[r.id][c.id];
      if (!cell) { tr.appendChild(el("td", null, el("div", { class: "cell none" }, r.id === "wg_train" && c.id === "Orig" ? "not used" : "none"))); return; }
      var shown = cell.use.filter(function (u) { return u !== "train" && u !== "not scored"; });
      tr.appendChild(el("td", null, el("button", {
        type: "button", class: "cell", "data-r": r.id, "data-c": c.id, "aria-pressed": "false",
        "data-src": M.src, "aria-label": r.name + ", " + c.label + ": " + fmt(cell.n) + " rows, used for " + cell.use.join(" and "),
        onclick: function () { sel = { r: r.id, c: c.id }; paint(); }
      }, [el("span", { class: "c" }, fmt(cell.n)), shown.length ? el("span", { class: "u" }, shown.map(function (u) { return el("span", { class: "use" }, u); })) : null])));
    });
    tbody.appendChild(tr);
  });
  table.appendChild(thead); table.appendChild(tbody);

  function exampleFor(r, c) {
    /* verified per-cell examples from the release (data/cell_examples_wg_*.json) come first */
    var ce = D.cellExamples && D.cellExamples[r + ":" + c];
    if (ce && ce.jb) return { jb: ce.jb, lang: ce.lang, src: ce.src, note: ce.note, id: ce.id };
    if (ce && ce.text) {
      return { text: ce.text, fills: ce.fills, arm: ce.arm === "en" ? "tr" : ce.arm, lang: ce.lang, gloss: ce.gloss,
               fillGloss: ce.lang !== "english" && ce.arm === "ph" ? fillGlossText(ce.fills) : null, src: ce.src, note: ce.note, id: ce.id, long: ce.text.length > 300 };
    }
    var b1 = (D.behaviors || []).filter(function (b) { return b.id === 1; })[0];
    var de = b1 && b1.byLang.german, en = b1 && b1.byLang.english;
    var wgw = (D.walks || []).filter(function (w) { return /wildguard/i.test(w.track || ""); });
    function vx(v, arm, lk) { if (!v) return null; return { text: v.text, fills: v.fills, arm: arm, lang: lk, gloss: v.gloss, src: "HarmBench, " + langName(lk) }; }
    if (r === "hb") {
      if (c === "Orig" && b1) return { text: b1.seed, fills: b1.seedSpans, arm: "seed", lang: "english", src: "HarmBench" };
      if (c === "EN" && en) return vx((en.variants.tr || [])[0], "tr", "english");
      if (c === "TR" && de) return vx((de.variants.tr || [])[0], "tr", "german");
      if (c === "PH" && de) return vx((de.variants.ph || [])[0], "ph", "german");
      if (c === "ORD") {
        /* a benign row on the same stored template as the harmful rows */
        var hitD = null;
        ["german", "ukrainian", "arabic", "japanese", "swahili", "english"].some(function (lk) {
          return (D.behaviors || []).some(function (b) {
            var d = b.byLang[lk]; if (!d) return false;
            var v = (d.variants.or || []).filter(function (x) { return x.cls === "direct" && (x.tk || "main") === "main"; })[0];
            if (v) { hitD = vx(v, "or", lk); hitD.src = "HarmBench, " + langName(lk); }
            return !!v;
          });
        });
        return hitD;
      }
      if (c === "ORB") {
        var hit = null;
        (D.behaviors || []).some(function (b) { return Object.keys(b.byLang).some(function (lk) { var v = (b.byLang[lk].variants.or || []).filter(function (x) { return x.cls === "bounded"; })[0]; if (v) { hit = vx(v, "or", lk); hit.src = "HarmBench, " + langName(lk); } return !!v; }); });
        return hit;
      }
      if (c === "JB" && D.jb && D.jb.length) { var j = D.jb[0]; return { jb: j, lang: j.language, src: "HB-JB (wrapped benign), " + langName(j.language) + ", wrapper: " + j.wrapper_name }; }
    }
    if (r === "wg_train" && wgw.length) {
      /* TR and OR from the same seed, so the two cells show a minimal pair; PH from the other worked example */
      var w0 = wgw[0], w1 = wgw[1] || wgw[0];
      var wx = function (w, text, fills, arm) { return { text: text, fills: fills, arm: arm, lang: w.lang, gloss: fillGlossText(fills) || null, src: "WildGuard train, " + langName(w.lang), long: text.length > 300 }; };
      if (c === "TR" && w1.stages[4]) return wx(w1, w1.stages[4].text, w1.stages[4].spans, "tr");
      if (c === "PH" && w0.stages[5]) return wx(w0, w0.stages[5].text, w0.stages[5].spans, "ph");
      if (c === "ORD") {
        var wo = [w1, w0].filter(function (w) { return w.orPair && w.orPair.cls === "direct"; })[0];
        if (wo) return wx(wo, wo.orPair.text, wo.orPair.fills, "or");
      }
    }
    return null;
  }
  function paint() {
    Array.prototype.forEach.call(table.querySelectorAll(".cell[data-r]"), function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-r") === sel.r && b.getAttribute("data-c") === sel.c));
    });
    var col = COLS.filter(function (c) { return c.id === sel.c; })[0];
    var row = M.rows.filter(function (r) { return r.id === sel.r; })[0];
    var cell = M.cells[sel.r][sel.c], def = M.arms[col.def] || {};
    detail.textContent = "";
    var uses = { train: "training (SFT, and the DPO prompt pool)", ASR: "attack success evaluation", OR: "over-refusal evaluation" };
    detail.appendChild(el("div", null, [
      paneLabel(row.name + ", " + col.label + ", " + fmt(cell.n) + " rows", col.arm),
      el("p", { style: "font-size:14.5px" }, [el("strong", null, def.label ? def.label + ". " : ""), cell.what || def.what || ""]),
      el("p", { class: "faint", style: "font-size:14.5px;margin-top:8px" }, cell.use[0] === "not scored" ? "Released but not scored: BOUNDED pairs never count toward over-refusal." : "Used for " + cell.use.map(function (u) { return uses[u] || u; }).join("; ") + ".")
    ]));
    var ex = exampleFor(sel.r, sel.c);
    /* every cell with rows has an example; this only shows if a build was made without the example files */
    if (!ex) { detail.appendChild(empty("Example not in this build", "This copy of the page was built without an example row for this cell.")); return; }
    function note(text) { return text ? el("p", { class: "faint ex-note" }, text) : null; }
    function gl(text) { var g = glossLine(text); if (g && text.length > 300) g.classList.add("long"); return g; }
    var t = setLang(el("div", { class: "pane-text ex" + (ex.long || ex.jb ? " long" : "") }), ex.lang);
    if (ex.jb) {
      var j = ex.jb, s = j.display_text;
      t.style.whiteSpace = "pre-wrap"; t.style.fontSize = "14px";
      var parts = [];
      if (j.truncated && j.marker_start != null) parts.push({ start: j.marker_start, end: j.marker_end, slot: null, arm: "en" });
      parts.push({ start: j.request_start, end: j.request_end, slot: "request", arm: "or" });
      var pos = 0;
      parts.sort(function (a, b) { return a.start - b.start; }).forEach(function (p) {
        t.appendChild(document.createTextNode(s.slice(pos, p.start)));
        t.appendChild(p.slot ? el("mark", { class: "fill", "data-arm": "or", "data-slot": "request" }, s.slice(p.start, p.end)) : el("span", { class: "faint", lang: "en" }, s.slice(p.start, p.end)));
        pos = p.end;
      });
      t.appendChild(document.createTextNode(s.slice(pos)));
      detail.appendChild(el("div", { class: "pane", "data-src": ex.id || null }, [paneLabel("Example, " + ex.src, "jb"), t, gl("Wrapper: " + j.gloss_wrapper), gl("Request: " + j.gloss_request),
        note(ex.note || "Harmful wrapped rows use the same wrappers around PH requests; this page does not print one.")]));
    } else {
      t.appendChild(filledFrag(ex.text, ex.fills, ex.arm));
      var en = ex.lang === "english";
      /* full translation (a long one scrolls in its own box); a PH row also glosses its localized entities */
      var gls = en ? [] : ex.fillGloss && ex.gloss ? [glossLine(ex.fillGloss), gl(ex.gloss)] : [gl(ex.gloss)];
      detail.appendChild(el("div", { class: "pane", "data-src": ex.id || null }, [paneLabel("Example, " + ex.src, ex.arm === "seed" ? "en" : ex.arm), t].concat(gls,
        [note(ex.arm === "seed" && ex.fills && ex.fills.length ? "Marked: the spans the English template turns into slots." : null), note(ex.note)])));
    }
    var sp = detail.querySelector(".pane-text.long mark");
    if (sp) scrollToMark(sp.closest(".pane-text"), sp);
    /* a wrapped row: keep the last lines of the wrapper in view above the request */
    if (sp && ex.jb) { var bx = sp.closest(".pane-text"), lh = parseFloat(getComputedStyle(bx).lineHeight) || 24; bx.scrollTop = Math.max(0, bx.scrollTop - 3 * lh); }
  }
  paint();
})();

/* ---------------- charts ---------------- */
var charts = [];
function chart(container, draw) {
  var c = { el: container, draw: draw };
  charts.push(c);
  return c;
}
function redraw(c) {
  var w = Math.max(c.minW || 280, Math.floor(c.el.clientWidth || 600));
  c.el.textContent = "";
  c.el.appendChild(c.draw(w));
}
function redrawAll() { charts.forEach(redraw); }

/* Finding 1: dot plot Orig / TR / PH per model */
(function () {
  var rows = (D.frontier && D.frontier.rows) || [];
  var track = "wg";
  var seg = $("#f1-seg"), legend = $("#f1-legend"), note = $("#f1-track-note");
  var NOTE = {
    hb: "",
    wg: "On the held-out WildGuard test split, translated variants break Olmo, Gemma and both Apertus models 2.4 to 4.4 times as often as the original English seeds. Llama, as on HarmBench, fails most often in English."
  };
  function setSeg() {
    segment(seg, [{ id: "wg", label: "WildGuard test (500 seeds, 44 langs)" }, { id: "hb", label: "HarmBench (200 seeds, 60 langs)" }], track, function (t) { track = t; setSeg(); redraw(c); });
    note.textContent = NOTE[track]; note.hidden = !NOTE[track];
  }
  setSeg();
  legend.appendChild(el("span", null, [el("i", { class: "dot", style: "background:var(--en)" }), "Orig. (English seed)"]));
  legend.appendChild(el("span", null, [el("i", { class: "dot", style: "background:var(--tr)" }), "TR translated"]));
  legend.appendChild(el("span", null, [el("i", { class: "dot", style: "background:var(--ph)" }), "PH localized"]));
  legend.appendChild(el("span", null, [el("i", { style: "height:2px;width:16px;background:var(--rule-2)" }), "range across the three"]));
  var c = chart($("#f1-chart"), function (W) {
    var narrow = W < 600;
    var labW = narrow ? 0 : 196, rowH = narrow ? 56 : 44, top = 26, right = narrow ? 14 : 150, bottom = 30;
    var H = top + rows.length * rowH + bottom;
    var x0 = labW + (narrow ? 6 : 14), x1 = W - right, max = 15;
    function X(v) { return x0 + (x1 - x0) * Math.min(v, max) / max; }
    var s = svg("svg", { class: "viz", viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "Attack success by model: original English, translated and localized prompts" });
    var ink = C("ink"), faint = C("faint"), rule = C("rule"), rule2 = C("rule-2"), tint = C("tint"), paper = C("paper");
    [0, 5, 10, 15].forEach(function (t) {
      s.appendChild(svg("line", { x1: X(t), x2: X(t), y1: top - 8, y2: H - bottom + 4, stroke: rule, "stroke-width": 1 }));
      s.appendChild(svg("text", { x: X(t), y: H - 10, "text-anchor": t === 0 ? "start" : (t === max && narrow ? "end" : "middle"), "font-size": 12, fill: faint }, t + "%"));
    });
    if (!narrow) {
      [["ORIG", "en"], ["TR", "tr"], ["PH", "ph"]].forEach(function (h, k) {
        s.appendChild(svg("text", { x: x1 + 44 + k * 44, y: 14, "text-anchor": "end", "font-size": 12, fill: C(h[1]), "font-weight": 500 }, h[0]));
      });
    }
    rows.forEach(function (r, k) {
      var o = num(r[track + "_orig"]), t = num(r[track + "_tr"]), p = num(r[track + "_ph"]);
      var yc = top + k * rowH + (narrow ? 36 : rowH / 2);
      var hi = /Olmo/.test(r.model) && track === "hb";
      if (hi) s.appendChild(svg("rect", { x: 0, y: top + k * rowH + 2, width: W, height: rowH - 4, fill: tint }));
      if (narrow) s.appendChild(svg("text", { x: x0, y: top + k * rowH + 16, "font-size": 13, "font-weight": hi ? 600 : 400, fill: ink }, r.model));
      else s.appendChild(svg("text", { x: labW, y: yc + 4, "text-anchor": "end", "font-size": 13, "font-weight": hi ? 600 : 400, fill: ink }, r.model));
      var lo = Math.min(o, t, p), hiV = Math.max(o, t, p);
      s.appendChild(svg("line", { x1: X(lo), x2: X(hiV), y1: yc, y2: yc, stroke: rule2, "stroke-width": 2 }));
      [[p, "ph", "PH"], [t, "tr", "TR"], [o, "en", "Orig"]].forEach(function (d) {
        s.appendChild(svg("circle", { cx: X(d[0]), cy: yc, r: 5.5, fill: C(d[1]), stroke: paper, "stroke-width": 1.5 }, svg("title", null, r.model + ", " + d[2] + ": " + pct(r[track + "_" + (d[1] === "en" ? "orig" : d[1])]) + "%")));
      });
      if (!narrow) {
        [["orig", "en"], ["tr", "tr"], ["ph", "ph"]].forEach(function (h, j) {
          s.appendChild(svg("text", { x: x1 + 44 + j * 44, y: yc + 4, "text-anchor": "end", "font-size": 13, "font-weight": hi && j < 2 ? 600 : 400, fill: C(h[1]) }, pct(r[track + "_" + h[0]])));
        });
      } else {
        var tx = svg("text", { x: x1, y: top + k * rowH + 16, "text-anchor": "end", "font-size": 12, fill: faint });
        tx.appendChild(svg("tspan", { fill: C("en") }, pct(r[track + "_orig"])));
        tx.appendChild(svg("tspan", { fill: C("ink-2") }, " / "));
        tx.appendChild(svg("tspan", { fill: C("tr") }, pct(r[track + "_tr"])));
        tx.appendChild(svg("tspan", { fill: C("ink-2") }, " / "));
        tx.appendChild(svg("tspan", { fill: C("ph") }, pct(r[track + "_ph"])));
        s.appendChild(tx);
      }
    });
    return s;
  });
  redraw(c);
})();

/* Finding 2: per-language TR vs PH bars */
(function () {
  var P = D.posthoc;
  if (!P) return;
  var model = P.models.filter(function (m) { return /Olmo/.test(m); })[0] || P.models[0];
  var seg = $("#f2-seg"), legend = $("#f2-legend");
  function setSeg() { segment(seg, P.models.map(function (m) { return { id: m, label: m }; }), model, function (m) { model = m; setSeg(); redraw(c); }); }
  setSeg();
  legend.appendChild(el("span", null, [el("i", { style: "background:var(--en)" }), "EN English template"]));
  legend.appendChild(el("span", null, [el("i", { style: "background:var(--tr)" }), "TR translated"]));
  legend.appendChild(el("span", null, [el("i", { style: "background:var(--ph)" }), "PH localized"]));
  var maxV = 0;
  P.languages.forEach(function (l) { P.models.forEach(function (m) { var d = P.data[l][m]; maxV = Math.max(maxV, d.TR || 0, d.PH || 0); }); });
  var top = Math.ceil(maxV / 10) * 10;
  var c = chart($("#f2-chart"), function (W) {
    var narrow = W < 600, padL = 34, padR = 6, padT = 24, padB = narrow ? 62 : 72, plotH = narrow ? 190 : 220, H = padT + plotH + padB;
    var langs = P.languages, n = langs.length, gap = narrow ? 6 : 14, sep = narrow ? 10 : 22;
    var inner = W - padL - padR - sep * 2;
    var gw = (inner - gap * (n - 1)) / n, bw = Math.min(26, (gw - 2) / 2);
    var Y = function (v) { return padT + plotH * (1 - v / top); };
    var s = svg("svg", { class: "viz", viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "HarmBench attack success per language, translated vs localized, for " + model });
    var faint = C("faint"), rule = C("rule"), ink = C("ink"), tr = C("tr"), ph = C("ph"), en = C("en");
    for (var t = 0; t <= top; t += 10) {
      s.appendChild(svg("line", { x1: padL, x2: W - padR, y1: Y(t), y2: Y(t), stroke: rule, "stroke-width": 1 }));
      s.appendChild(svg("text", { x: padL - 6, y: Y(t) + 4, "text-anchor": "end", "font-size": 12, fill: faint }, t + "%"));
    }
    var x = padL;
    var groupOf = function (l) { return l === "english" ? 0 : (P.groups["lower-resource"].indexOf(l) >= 0 ? 1 : 2); };
    var groupStart = {}, groupEnd = {};
    langs.forEach(function (l, i) {
      var gidx = groupOf(l);
      if (i > 0 && gidx !== groupOf(langs[i - 1])) x += sep;
      var d = P.data[l][model], gx = x + (gw - (d.PH == null ? bw : bw * 2 + 2)) / 2;
      if (groupStart[gidx] == null) groupStart[gidx] = x; groupEnd[gidx] = x + gw;
      var bars = [{ v: d.TR, c: l === "english" ? en : tr, k: l === "english" ? "EN" : "TR" }]; if (d.PH != null) bars.push({ v: d.PH, c: ph, k: "PH" });
      bars.forEach(function (b, j) {
        var bx = gx + j * (bw + 2), y = Y(b.v);
        s.appendChild(svg("rect", { x: bx, y: y, width: bw, height: Math.max(0.5, Y(0) - y), fill: b.c }, svg("title", null, langName(l) + " " + b.k + ": " + b.v.toFixed(1) + "%")));
        s.appendChild(svg("text", { x: bx + bw / 2, y: y - 4, "text-anchor": "middle", "font-size": 12, fill: ink }, narrow ? Math.round(b.v) : b.v.toFixed(1)));
      });
      var yb = padT + plotH;
      s.appendChild(svg("text", { x: x + gw / 2, y: yb + 16, "text-anchor": "middle", "font-size": 12, fill: ink, "font-weight": 500 }, iso(l)));
      if (!narrow) s.appendChild(svg("text", { x: x + gw / 2, y: yb + 32, "text-anchor": "middle", "font-size": 12, fill: faint }, l === "english" ? "EN only" : langName(l)));
      x += gw + gap;
    });
    var gl = ["English", "lower-resource", "higher-resource"];
    var yLine = padT + plotH + (narrow ? 28 : 44);
    [1, 2].forEach(function (gi) {
      if (groupStart[gi] == null) return;
      s.appendChild(svg("line", { x1: groupStart[gi], x2: groupEnd[gi], y1: yLine, y2: yLine, stroke: C("rule-2"), "stroke-width": 1 }));
      s.appendChild(svg("text", { x: (groupStart[gi] + groupEnd[gi]) / 2, y: yLine + 16, "text-anchor": "middle", "font-size": 12, fill: faint }, gl[gi]));
    });
    if (narrow) s.appendChild(svg("text", { x: groupStart[0] + gw / 2, y: yLine + 16, "text-anchor": "middle", "font-size": 12, fill: faint }, "EN"));
    return s;
  });
  redraw(c);
})();

/* Finding 3: ASR vs OR scatter */
(function () {
  var rows = (D.frontier && D.frontier.rows) || [];
  var view = "hb";
  var V = {
    hb: { x: "hb_tr", y: "hb_or", xl: "HarmBench ASR, mean TR (%)", yl: "HarmBench OR (%)", cap: "" },
    wg: { x: "wg_tr", y: "wg_or", xl: "WildGuard test ASR, mean TR (%)", yl: "WildGuard test OR (%)", cap: "" },
    jb: { x: "hbjb_asr", y: "hbjb_or", xl: "HB-JB ASR on wrapped PH (%)", yl: "HB-JB OR on wrapped benign (%)", cap: "HB-JB: jailbreak-wrapped HarmBench, 300 prompts per language. ASR on wrapped PH requests, OR on wrapped benign prompts. Qwen3.5-27B's ASR is below 0.1% and is drawn at 0.05." }
  };
  var seg = $("#f3-seg");
  function setSeg() { segment(seg, [{ id: "hb", label: "HarmBench" }, { id: "wg", label: "WildGuard test" }, { id: "jb", label: "HB-JB (wrapped)" }], view, function (v) { view = v; setSeg(); $("#f3-cap").textContent = V[v].cap; redraw(c); }); }
  setSeg();
  $("#f3-cap").textContent = V[view].cap;
  function nice(m) { var steps = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]; for (var i = 0; i < steps.length; i++) if (steps[i] >= m) return steps[i]; return Math.ceil(m / 10) * 10; }
  var c = chart($("#f3-chart"), function (W) {
    var v = V[view], narrow = W < 600;
    var H = narrow ? 340 : 400, padL = 46, padR = narrow ? 12 : 230, padT = 16, padB = 46;
    var pts = rows.map(function (r) { return { m: r.model, x: num(r[v.x]), y: num(r[v.y]), xs: pct(r[v.x]), ys: pct(r[v.y]) }; });
    var mx = nice(Math.max.apply(null, pts.map(function (p) { return p.x; })) * 1.08), my = nice(Math.max.apply(null, pts.map(function (p) { return p.y; })) * 1.08);
    var X = function (q) { return padL + (W - padL - padR) * q / mx; }, Y = function (q) { return padT + (H - padT - padB) * (1 - q / my); };
    var s = svg("svg", { class: "viz", viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "Attack success vs over-refusal per model" });
    var faint = C("faint"), rule = C("rule"), ink = C("ink"), tint = C("tint"), good = C("or");
    // ideal corner
    s.appendChild(svg("rect", { x: X(0), y: Y(my * 0.2), width: X(mx * 0.2) - X(0), height: Y(0) - Y(my * 0.2), fill: tint }));
    s.appendChild(svg("text", { x: padL, y: H - 8, "font-size": 12, fill: good, "font-weight": 500 }, "\u2199 shaded: low ASR and low OR"));
    var stepX = mx <= 15 ? 5 : 10, stepY = my <= 20 ? 5 : 10;
    for (var t = 0; t <= mx + 1e-9; t += stepX) {
      s.appendChild(svg("line", { x1: X(t), x2: X(t), y1: padT, y2: Y(0), stroke: rule, "stroke-width": 1 }));
      s.appendChild(svg("text", { x: X(t), y: Y(0) + 16, "text-anchor": "middle", "font-size": 12, fill: faint }, t));
    }
    for (var u = 0; u <= my + 1e-9; u += stepY) {
      s.appendChild(svg("line", { x1: padL, x2: W - padR, y1: Y(u), y2: Y(u), stroke: rule, "stroke-width": 1 }));
      s.appendChild(svg("text", { x: padL - 8, y: Y(u) + 4, "text-anchor": "end", "font-size": 12, fill: faint }, u));
    }
    s.appendChild(svg("text", { x: W - padR, y: H - 8, "text-anchor": "end", "font-size": 12, fill: C("tr") }, (narrow ? "ASR" : v.xl) + " →"));
    s.appendChild(svg("text", { x: 12, y: (padT + H - padB) / 2, "text-anchor": "middle", "font-size": 12, fill: C("or"), transform: "rotate(-90 12 " + (padT + H - padB) / 2 + ")" }, v.yl + " →"));
    // labels: greedy placement next to each dot, avoiding other dots and labels; a leader line when moved off the dot
    var P2 = pts.map(function (p, k) { return { p: p, k: k + 1, x: X(p.x), y: Y(p.y) }; });
    P2.forEach(function (L) { s.appendChild(svg("circle", { cx: L.x, cy: L.y, r: 5.5, fill: ink, stroke: C("paper"), "stroke-width": 1.5 }, svg("title", null, L.p.m + ": ASR " + L.p.xs + "%, OR " + L.p.ys + "%"))); });
    var fs = narrow ? 12 : 13, cw = fs * 0.6;
    var boxes = P2.map(function (L) { return { x0: L.x - 7, x1: L.x + 7, y0: L.y - 7, y1: L.y + 7 }; });
    function hit(b) {
      if (b.x0 < padL - 2 || b.x1 > W - 2 || b.y0 < 0 || b.y1 > Y(0) - 2) return 1e6;
      var o = 0;
      boxes.forEach(function (q) { var w = Math.min(b.x1, q.x1) - Math.max(b.x0, q.x0), h = Math.min(b.y1, q.y1) - Math.max(b.y0, q.y0); if (w > 0 && h > 0) o += w * h; });
      return o;
    }
    var CAND = [[10, 0], [-10, 0], [8, -14], [8, 14], [-8, -14], [-8, 14], [12, -28], [12, 28], [-12, -28], [-12, 28], [14, -44], [14, 44]];
    P2.slice().sort(function (a, b) { return a.x - b.x || a.y - b.y; }).forEach(function (L) {
      var name = narrow ? String(L.k) : L.p.m, val = narrow ? "" : " " + L.p.xs + " / " + L.p.ys, wd = (name.length + val.length) * cw + 4, best = null;
      CAND.forEach(function (cd) {
        var right = cd[0] > 0, bx0 = right ? L.x + cd[0] : L.x + cd[0] - wd, b = { x0: bx0, x1: bx0 + wd, y0: L.y + cd[1] - fs * 0.62, y1: L.y + cd[1] + fs * 0.5 };
        var sc = hit(b) + Math.abs(cd[1]) * 0.6 + (right ? 0 : 4);
        if (!best || sc < best.sc) best = { sc: sc, b: b, cd: cd, right: right };
      });
      boxes.push(best.b);
      var ty = L.y + best.cd[1] + fs * 0.36;
      if (Math.abs(best.cd[1]) >= 14) s.appendChild(svg("line", { x1: L.x, y1: L.y + (best.cd[1] > 0 ? 6 : -6), x2: best.right ? best.b.x0 - 1 : best.b.x1 + 1, y2: L.y + best.cd[1] - 2, stroke: C("rule-2"), "stroke-width": 1 }));
      var key = /gpt-oss|Qwen/.test(L.p.m);
      var t = svg("text", { x: best.right ? best.b.x0 : best.b.x1, y: ty, "text-anchor": best.right ? "start" : "end", "font-size": fs, fill: ink, "font-weight": key ? 600 : 400, stroke: C("paper"), "stroke-width": 3, "paint-order": "stroke" });
      t.appendChild(svg("tspan", null, name));
      if (val) t.appendChild(svg("tspan", { fill: faint, "font-weight": 400 }, val));
      s.appendChild(t);
    });
    var wrapEl = document.createElement("div");
    wrapEl.appendChild(s);
    if (narrow) {
      wrapEl.appendChild(el("ul", { class: "pt-list", "aria-label": "Values per model, ASR / OR" }, pts.map(function (p) {
        return el("li", null, [el("span", null, [el("b", { class: "num", style: "font-weight:600;margin-right:6px" }, String(pts.indexOf(p) + 1)), p.m]), el("span", null, p.xs + " / " + p.ys)]);
      })));
    }
    return wrapEl;
  });
  redraw(c);
})();

/* ---------------- training ---------------- */
(function () {
  var T = D.training;
  if (!T) return;
  var rows = T.rows;
  var FAM = [
    { id: "a-base", label: "Apertus-8B base", fam: "Apertus-8B-2509", start: "Base" },
    { id: "a-it", label: "Apertus-8B-Instruct", fam: "Apertus-8B-2509", start: "IT" },
    { id: "a15", label: "Apertus-v1.5-8B-Instruct", fam: "Apertus-v1.5-8B", start: "IT" },
    { id: "olmo", label: "Olmo-3-7B-Instruct", fam: "Olmo-3-7B", start: "IT" },
    { id: "llama", label: "Llama-3.1-8B-Instruct", fam: "Llama-3.1-8B", start: "IT" }
  ];
  var METRICS = [
    { g: "Safety", sub: "attack success / toxicity, lower is better", items: [["AyaRT", "AyaRT"], ["PolyTox", "PolyTox"], ["HB_ASR", "HB ASR"], ["WG_ASR", "WG ASR"]], dir: -1 },
    { g: "Over-refusal", sub: "lower is better", items: [["HB_OR", "HB OR"], ["WG_OR", "WG OR"], ["OR-Bench", "OR-Bench"]], dir: -1 }
  ];
  var fam = "a-it";
  function pathRows(f) {
    var F = FAM.filter(function (x) { return x.id === f; })[0];
    var p = [F.start, F.start + " + SFT", F.start + " + SFT + DPO"];
    return p.map(function (path) { return rows.filter(function (r) { return r.family === F.fam && r.path === path; })[0]; });
  }
  var maxOf = {};
  METRICS.forEach(function (g) { g.items.forEach(function (it) { maxOf[it[0]] = Math.max.apply(null, rows.map(function (r) { return r[it[0]] || 0; })); }); });
  var seg = $("#fam-seg"), box = $("#multiples");
  function setSeg() { segment(seg, FAM.map(function (f) { return { id: f.id, label: f.label }; }), fam, function (f) { fam = f; setSeg(); build(); }); }
  function build() {
    box.textContent = ""; charts = charts.filter(function (c) { return !c.mult; });
    var pr = pathRows(fam);
    METRICS.forEach(function (g) {
      var grp = el("div", { class: "mrow" });
      grp.appendChild(el("div", { class: "mgroup" }, [el("h3", null, g.g), el("span", { class: "faint", style: "font-size:13px" }, g.sub)]));
      box.appendChild(grp);
      g.items.forEach(function (it) {
        var key = it[0], cell = el("div", { class: "mcell" });
        cell.appendChild(el("div", { class: "mt" }, [el("b", null, it[1]), el("span", { class: "dir" }, g.dir < 0 ? "↓" : "↑")]));
        var holder = el("div");
        cell.appendChild(holder);
        grp.appendChild(cell);
        var c = chart(holder, function (W) { return mini(W, pr, key, g.dir); });
        c.mult = true; c.minW = 130;
        redraw(c);
      });
    });
    var dag = pr[0] && pr[0].flags && Object.keys(pr[0].flags).some(function (k) { return pr[0].flags[k].indexOf("dagger") >= 0; });
    $("#mult-cap").textContent = dag ? "† Base checkpoint served with the instruct chat template; most of its replies are degenerate and only the remaining ones are scored." : "";
  }
  function mini(W, pr, key, dir) {
    var H = 132, padT = 22, padB = 34, padL = 4, padR = 4;
    var vals = pr.map(function (r) { return r ? r[key] : null; }).filter(function (x) { return x != null; });
    var top = Math.max(5, Math.max.apply(null, vals) * 1.15);
    var max = top <= 50 ? Math.ceil(top / 5) * 5 : Math.ceil(top / 10) * 10;
    var narrowM = W < 220;
    var Y = function (v) { return padT + (H - padT - padB) * (1 - v / max); };
    var s = svg("svg", { class: "viz", viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img" });
    var ink = C("ink"), ink2 = C("ink-2"), faint = C("faint"), rule = C("rule"), paper = C("paper"), good = C("good"), bad = C("bad");
    s.appendChild(svg("line", { x1: padL, x2: W - padR, y1: Y(0), y2: Y(0), stroke: C("rule-2"), "stroke-width": 1 }));
    var n = 3, slot = (W - padL - padR) / n, bw = Math.min(54, slot * 0.5);
    var start = pr[0] && pr[0][key];
    if (start != null) s.appendChild(svg("line", { x1: padL, x2: W - padR, y1: Y(start), y2: Y(start), stroke: ink2, "stroke-width": 1, "stroke-dasharray": "3 3" }));
    var labels = ["start", "+SFT", "+DPO"];
    var aria = [];
    pr.forEach(function (r, i) {
      if (!r) return;
      var val = r[key]; if (val == null) return;
      var x = padL + slot * i + (slot - bw) / 2, y = Y(val);
      var fill = i === 0 ? paper : (i === 1 ? faint : ink2);
      s.appendChild(svg("rect", { x: x, y: y, width: bw, height: Math.max(0.75, Y(0) - y), fill: fill, stroke: i === 0 ? ink2 : "none", "stroke-width": i === 0 ? 1.25 : 0 }));
      var dag = r.flags && r.flags[key] && r.flags[key].indexOf("dagger") >= 0;
      var ly = y - 6;
      if (i > 0 && start != null) { var ls = Y(start); if (ls > ly - 12 && ls < ly + 4) ly = ls - 5; }
      s.appendChild(svg("text", { x: x + bw / 2, y: ly, "text-anchor": "middle", "font-size": narrowM ? 12 : 13, "font-weight": i === 2 ? 600 : 400, fill: ink, stroke: paper, "stroke-width": 4, "paint-order": "stroke", "stroke-linejoin": "round" }, val.toFixed(1) + (dag ? "†" : "")));
      s.appendChild(svg("text", { x: x + bw / 2, y: H - padB + 15, "text-anchor": "middle", "font-size": 12, fill: faint, "font-weight": 500 }, labels[i]));
      if (i > 0 && start != null) {
        var d = Math.round((val - start) * 10) / 10, better = d * dir > 0, worse = d * dir < 0;
        s.appendChild(svg("text", { x: x + bw / 2, y: H - padB + 29, "text-anchor": "middle", "font-size": 12, "font-weight": 600, fill: better ? good : (worse ? bad : faint) }, (d > 0 ? "+" : d < 0 ? "−" : "±") + Math.abs(d).toFixed(1)));
      }
      aria.push(labels[i] + " " + val.toFixed(1));
    });
    s.setAttribute("aria-label", key + ": " + aria.join(", "));
    return s;
  }
  setSeg(); build();

  // generalization
  var G = T.gen;
  if (G) {
    $("#gen-text").textContent = "The WildGuard training track covers 44 languages and HarmBench evaluates 60. Of HarmBench's 59 non-English languages, 43 appear in the training data and " + G.heldOut.length + " never do. On those held-out languages (" + G.heldOut.join(", ") + "), attack success after MSafeTemp training ends at most 1.2 points above the trained languages.";
    var tb = el("tbody");
    var pairs = ["Apertus-8B-Instruct", "Olmo-3-7B-Instruct", "Llama-3.1-8B-Instruct"].map(function (m) { return [m, m + " + MSafeTemp"]; });
    function R(m) { return G.rows.filter(function (r) { return r.model === m; })[0]; }
    pairs.forEach(function (p) {
      var a = R(p[0]), b = R(p[1]); if (!a || !b) return;
      tb.appendChild(el("tr", null, [
        el("td", null, p[0]),
        el("td", { class: "r" }, pct(a.tr_seen)), el("td", { class: "r good" }, pct(b.tr_seen)),
        el("td", { class: "r" }, pct(a.tr_heldout)), el("td", { class: "r good" }, pct(b.tr_heldout))
      ]));
    });
    var gt = $("#gen-table");
    gt.appendChild(el("thead", null, [
      el("tr", null, [el("th", null, ""), el("th", { class: "r", colspan: "2" }, "43 trained languages"), el("th", { class: "r", colspan: "2" }, G.heldOut.length + " held-out languages")]),
      el("tr", null, [el("th", null, "Model"), el("th", { class: "r" }, "Start"), el("th", { class: "r" }, "After"), el("th", { class: "r" }, "Start"), el("th", { class: "r" }, "After")])
    ]));
    gt.appendChild(tb);
  }

})();

/* ---------------- cite ---------------- */
(function () {
  var pre = $("#bibtex"), btn = $("#copy-bib");
  pre.textContent = (D.meta && D.meta.bibtex) || "";
  btn.addEventListener("click", function () {
    var text = pre.textContent;
    function selectIt() { var r = document.createRange(); r.selectNodeContents(pre); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = "Selected, press Ctrl+C"; }
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { btn.textContent = "Copied"; setTimeout(function () { btn.textContent = "Copy"; }, 1800); }, selectIt);
      } else selectIt();
    } catch (e) { selectIt(); }
  });
  if (D.meta && D.meta.ack) $("#ack").textContent = "This work was " + D.meta.ack.charAt(0).toLowerCase() + D.meta.ack.slice(1);
})();

/* link from finding 2 to the German walkthrough */
$("#f2-walk-link").addEventListener("click", function () { Walk.show("german"); });

/* ---------------- brand: M<Safe>Temp rolls through other fills and settles on Safe ---------------- */
(function () {
  var slot = document.querySelector(".brand .bslot"); if (!slot) return;
  var WORDS = ["Harm", "Wild", "Toxi", "Safe"], HOLD = 700, running = false;
  function roll(word, last, done) {
    var old = slot.firstElementChild, node = el("span", { class: "tw" + (last ? " slow" : "") }, word);
    if (last) old.classList.add("slow");
    old.classList.add("out");
    setTimeout(function () {
      node.classList.add("in"); slot.replaceChild(node, old);
      requestAnimationFrame(function () { requestAnimationFrame(function () { node.classList.remove("in"); }); });
      setTimeout(function () { node.classList.remove("slow"); done(); }, last ? 760 : 320);
    }, last ? 300 : 160);
  }
  function run() {
    if (running || RM.matches) return;
    running = true;
    var i = 0;
    (function next() {
      roll(WORDS[i], i === WORDS.length - 1, function () { if (++i < WORDS.length) setTimeout(next, HOLD); else running = false; });
    })();
  }
  /* no roll on load: it plays on hover, and once on the first scroll down the page */
  slot.parentNode.addEventListener("mouseenter", run);
  function onScroll() { if (window.scrollY > 40) { window.removeEventListener("scroll", onScroll); run(); } }
  window.addEventListener("scroll", onScroll, { passive: true });
})();

/* ---------------- title as a template ---------------- */
/* "How I <alignment_verb> Your <target>": the title opens as the bare template,
   runs a few fills and lands on the real title. A click goes back to the template first,
   then runs the fills again, word to word; the last swap, to the real title, is slower.
   Each slot is a ⟨field⟩ as wide as its longest word, so the line never moves. */
var TitleT = (function () {
  var FILLS = {
    alignment_verb: ["Fine-Tuned", "Red-Teamed", "Post-Trained", "Safety-Tuned", "Evaluated"],
    target: ["LLM", "Assistant", "Checkpoint", "Agent", "Weights"]
  };
  var CHIP_EM = 0.72;                        /* chip font size from the CSS, in em of the title */
  var FIRST_RUN = 4, CLICK_RUN = 4;          /* fills per run, the last one is the real title */
  var WORD_HOLD = 700, OPEN_HOLD = 900;      /* pause between words; delay before the first run */
  var ROLL = { out: 160, in: 320 }, ROLL_LAST = { out: 300, in: 760 };   /* ms; match .tw and .tw.slow */
  var line = $("#title .tline");
  if (!line) return { fit: function () {} };
  var slots = Array.prototype.slice.call(line.querySelectorAll(".tslot"));
  var FINAL = slots.map(function (s) { return s.dataset.final; });
  var running = false;
  var chipEm = slots.map(function () { return CHIP_EM; });

  function word(i, w) { return el("span", { class: "tw" }, w); }
  function isChip(s) { return s.firstElementChild.classList.contains("chip"); }
  function pickAfter(i, prev) {
    var opts = FILLS[slots[i].dataset.slot].filter(function (w) { return w !== prev; });
    return opts[Math.floor(Math.random() * opts.length)];
  }
  function chip(s, i) { return el("span", { class: "tw chip", style: "font-size:" + chipEm[i] + "em" }, s.dataset.slot); }
  /* Roll the slot's content out and the new node in; the slot itself keeps its width. */
  function morph(s, node, last, done) {
    var old = s.firstElementChild, t = last ? ROLL_LAST : ROLL;
    if (last) { old.classList.add("slow"); node.classList.add("slow"); }
    old.classList.add("out");
    setTimeout(function () {
      node.classList.add("in");
      s.replaceChild(node, old);
      requestAnimationFrame(function () { requestAnimationFrame(function () { node.classList.remove("in"); }); });
      setTimeout(function () { node.classList.remove("slow"); if (done) done(); }, t.in);
    }, t.out);
  }
  /* Morph every slot at once; done when all have finished. */
  function all(make, last, done) {
    var pending = slots.length;
    slots.forEach(function (s, i) { morph(s, make(s, i), last, function () { if (--pending === 0) done(); }); });
  }
  /* Rounds in lockstep: both slots refill together, word to word. With viaTags the
     slots first turn back into their tags and hold there. */
  function run(fills, viaTags) {
    if (running || RM.matches) return;
    running = true;
    var words = slots.map(function (s, i) {
      var list = [], prev = isChip(s) ? FINAL[i] : s.textContent;
      for (var r = 0; r < fills - 1; r++) { prev = pickAfter(i, prev); list.push(prev); }
      list.push(FINAL[i]);
      return list;
    });
    var round = 0;
    function fill() {
      all(function (s, i) { return word(i, words[i][round]); }, round === fills - 1, function () {
        round++;
        if (round < fills) setTimeout(fill, WORD_HOLD); else running = false;
      });
    }
    if (viaTags && !slots.every(isChip)) all(chip, false, function () { setTimeout(fill, WORD_HOLD); });
    else fill();
  }
  function settle() { slots.forEach(function (s, i) { s.replaceChildren(word(i, FINAL[i])); }); }

  /* Size each field to its longest word (the real one included) plus its brackets. */
  function fit() {
    var probe = el("span", { class: "tw", style: "position:absolute;left:0;top:0;visibility:hidden;white-space:nowrap;width:auto" });
    line.appendChild(probe);
    var em = parseFloat(getComputedStyle(line).fontSize);
    function width(text) { probe.textContent = text; return probe.getBoundingClientRect().width; }
    slots.forEach(function (sl, i) {
      var inner = Math.max.apply(null, FILLS[sl.dataset.slot].concat(FINAL[i]).map(width)) + 0.05 * em;   /* slack for the roll's tilt */
      var cs = getComputedStyle(sl);
      sl.style.width = Math.ceil(inner + parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight)) + "px";
      /* a tag wider than the box shrinks to fit; otherwise it keeps the CSS size */
      probe.className = "tw chip"; probe.style.fontSize = CHIP_EM + "em";
      var cw = width(sl.dataset.slot);
      chipEm[i] = cw > inner ? Math.round(CHIP_EM * inner / cw * 1000) / 1000 : CHIP_EM;
      probe.className = "tw"; probe.style.fontSize = "";
    });
    /* both tags at the size of the smaller one, so they read as a pair */
    var chipMin = Math.min.apply(null, chipEm);
    slots.forEach(function (sl, i) {
      chipEm[i] = chipMin;
      var cur = sl.firstElementChild;
      if (cur && cur.classList.contains("chip")) cur.style.fontSize = chipMin + "em";
    });
    probe.remove();
  }

  line.addEventListener("click", function () { run(CLICK_RUN, true); });
  fit();
  if (RM.matches) settle();
  else {
    /* First run once the title is on screen and the tab is visible. */
    var opened = false;
    var open = function () {
      if (opened || document.hidden) return;
      opened = true;
      setTimeout(function () { run(FIRST_RUN); }, OPEN_HOLD);
    };
    var whenSeen = function () {
      if (!("IntersectionObserver" in window)) { open(); return; }
      var io = new IntersectionObserver(function (es) { if (es[0].isIntersecting) { io.disconnect(); open(); } });
      io.observe(line);
    };
    document.addEventListener("visibilitychange", function () { if (!opened && !document.hidden) whenSeen(); });
    if (!document.hidden) whenSeen();
  }
  if (RM.addEventListener) RM.addEventListener("change", function () { if (RM.matches && !running) settle(); });
  return { fit: fit };
})();
window.addEventListener("resize", function () { TitleT.fit(); });
if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { TitleT.fit(); });

/* ---------------- theme + resize re-render ---------------- */
new MutationObserver(function () { redrawAll(); }).observe(ROOT, { attributes: true, attributeFilter: ["data-theme"] });
var mq = window.matchMedia("(prefers-color-scheme: dark)");
if (mq.addEventListener) mq.addEventListener("change", function () { applyTheme(curTheme); redrawAll(); });

/* sideways scrollers: fade the right edge while more is hidden, plus a one-line hint */
function scrollHints() {
  Array.prototype.forEach.call(document.querySelectorAll(".scroll-x"), function (sx) {
    var more = sx.scrollWidth - sx.clientWidth > 2;
    sx.classList.toggle("has-more", more && sx.scrollLeft + sx.clientWidth < sx.scrollWidth - 2);
    var h = sx.nextElementSibling && sx.nextElementSibling.classList.contains("scroll-hint") ? sx.nextElementSibling : null;
    if (!h) { h = el("p", { class: "scroll-hint", "aria-hidden": "true" }, "Scroll sideways for all columns \u2192"); sx.parentNode.insertBefore(h, sx.nextSibling); }
    h.classList.toggle("on", more);
    if (!sx.hasAttribute("data-hint")) {
      sx.setAttribute("data-hint", "");
      sx.addEventListener("scroll", function () { sx.classList.toggle("has-more", sx.scrollLeft + sx.clientWidth < sx.scrollWidth - 2); }, { passive: true });
    }
  });
}
scrollHints();
Array.prototype.forEach.call(document.querySelectorAll("details"), function (d) { d.addEventListener("toggle", scrollHints); });
var lastW = window.innerWidth, rz = null;
window.addEventListener("resize", function () {
  if (rz) cancelAnimationFrame(rz);
  rz = requestAnimationFrame(function () { if (window.innerWidth !== lastW) { lastW = window.innerWidth; redrawAll(); } scrollHints(); });
});
if (document.fonts && document.fonts.ready) document.fonts.ready.then(redrawAll);
})();
