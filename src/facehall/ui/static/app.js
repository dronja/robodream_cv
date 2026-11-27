// facehall ui: the page of `facehall ui` (D41). Plain JavaScript, no framework or CDN.
// File names and paths come from the disk, so they only ever go into textContent or attributes.
"use strict";

(() => {
  const SVG = "http://www.w3.org/2000/svg";
  const STORE = "facehall.ui.";
  const ACTIVE = ["queued", "loading", "running"];
  const DEFAULTS = {
    scale: 4,
    image_format: "png",
    video_format: "mp4",
    quality: 95,
    watermark: false,
    overwrite: false,
    subfolder: true,
    output: "",
  };

  const $ = (id) => document.getElementById(id);
  const store = {
    get(key, fallback) {
      try {
        const value = localStorage.getItem(STORE + key);
        return value === null ? fallback : JSON.parse(value);
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(STORE + key, JSON.stringify(value));
      } catch {
        /* private window or blocked storage: settings are not remembered */
      }
    },
  };

  let token = null;
  let info = null;
  let strings = { en: {} };
  let lang = "en";
  let theme = store.get("theme", "auto");
  const state = {
    media: [],
    settings: { ...DEFAULTS },
    batches: new Map(),
    cards: new Map(),
    submitting: false,
  };

  // --- text --------------------------------------------------------------------------------

  function t(key, params = {}) {
    let text = strings[lang]?.[key] ?? strings.en?.[key] ?? key;
    if (typeof text === "object") {
      const rule = new Intl.PluralRules(lang).select(Number(params.n ?? 0));
      text = text[rule] ?? text.other;
    }
    return text.replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match));
  }

  function number(value, digits = 0) {
    return Number(value).toLocaleString(lang, { maximumFractionDigits: digits });
  }

  function bytes(size) {
    const units = ["B", "kB", "MB", "GB", "TB"];
    let value = size;
    let unit = 0;
    while (value >= 1000 && unit < units.length - 1) {
      value /= 1000;
      unit += 1;
    }
    return `${number(value, unit ? 1 : 0)} ${units[unit]}`;
  }

  function basename(path) {
    const parts = String(path).replace(/[\\/]+$/, "").split(/[\\/]/);
    return parts[parts.length - 1] || path;
  }

  // Paths under the home folder are shown as ~/…; the service expands ~ again.
  function tilde(path) {
    const home = info?.home;
    const text = String(path);
    if (!home || home === "/") return text;
    if (text === home) return "~";
    return text.startsWith(`${home}/`) ? `~${text.slice(home.length)}` : text;
  }

  // The same inside a message of the service, which names full paths.
  function shorten(text) {
    const home = info?.home;
    if (!text || !home || home === "/") return text || "";
    return String(text).split(`${home}/`).join("~/");
  }

  function dirname(path) {
    const text = String(path).replace(/[\\/]+$/, "");
    const cut = Math.max(text.lastIndexOf("/"), text.lastIndexOf("\\"));
    return cut > 0 ? text.slice(0, cut) : "/";
  }

  // --- DOM ---------------------------------------------------------------------------------

  function h(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") el.className = value;
      else if (key === "text") el.textContent = value;
      else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  function icon(name, cls = "icon") {
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("class", cls);
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG, "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }

  function iconButton(name, label, onclick, cls = "icon-button") {
    return h("button", { class: cls, type: "button", title: label, "aria-label": label, onclick }, icon(name));
  }

  let snackTimer = null;
  function snackbar(text) {
    $("snackbar-text").textContent = text;
    $("snackbar").hidden = false;
    clearTimeout(snackTimer);
    snackTimer = setTimeout(() => {
      $("snackbar").hidden = true;
    }, 6000);
  }

  // --- the API -----------------------------------------------------------------------------

  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.status = status;
    }
  }

  async function api(path, options = {}) {
    const init = { method: options.method || "GET", headers: { "X-Facehall-Token": token || "" } };
    if (options.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    let response;
    try {
      response = await fetch(path, init);
    } catch {
      throw new ApiError(t("error.network"), 0);
    }
    if (response.status === 401) {
      lock();
      throw new ApiError(t("locked.title"), 401);
    }
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        if (typeof body.detail === "string") detail = body.detail;
        else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
      } catch {
        /* not JSON */
      }
      throw new ApiError(detail, response.status);
    }
    return response.status === 204 ? null : response.json();
  }

  function query(path, params) {
    return `${path}?${new URLSearchParams({ ...params, token: token || "" })}`;
  }

  const thumbnail = (path) => query("/api/thumbnail", { path });

  function fail(error) {
    if (error.status !== 401) snackbar(t("error.prefix", { message: shorten(error.message) }));
  }

  // --- the session key -----------------------------------------------------------------------

  function readToken() {
    const match = /(?:^#|&)token=([^&]+)/.exec(location.hash);
    if (match) {
      token = decodeURIComponent(match[1]);
      try {
        sessionStorage.setItem(STORE + "token", token);
      } catch {
        /* kept for this page only */
      }
      history.replaceState(null, "", location.pathname + location.search);
      return;
    }
    try {
      token = sessionStorage.getItem(STORE + "token");
    } catch {
      token = null;
    }
  }

  function lock() {
    $("main").hidden = true;
    $("locked").hidden = false;
    clearTimeout(pollTimer);
  }

  $("locked-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = $("locked-input").value.trim();
    const match = /token=([^&\s]+)/.exec(text);
    token = match ? decodeURIComponent(match[1]) : text;
    try {
      sessionStorage.setItem(STORE + "token", token);
    } catch {
      /* kept for this page only */
    }
    start();
  });

  // --- language and theme ------------------------------------------------------------------

  function applyStrings() {
    document.documentElement.lang = lang;
    for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
    for (const el of document.querySelectorAll("[data-i18n-title]")) {
      const label = t(el.dataset.i18nTitle);
      el.title = label;
      el.setAttribute("aria-label", label);
    }
    $("language-code").textContent = lang.toUpperCase();
    applyTheme();
    if (info) {
      renderMedia();
      renderSettings();
      for (const card of state.cards.values()) card.el.remove();
      state.cards.clear();
      renderBatches();
      if ($("viewer").open) renderViewer();
      if ($("picker").open) renderPicker();
    }
  }

  function applyTheme() {
    if (theme === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    $("theme-icon").setAttribute("href", { auto: "#i-contrast", light: "#i-light", dark: "#i-dark" }[theme]);
    const label = t("action.theme", { mode: t(`theme.${theme}`) });
    $("theme").title = label;
    $("theme").setAttribute("aria-label", label);
  }

  $("language").addEventListener("click", () => {
    lang = lang === "en" ? "uk" : "en";
    store.set("lang", lang);
    applyStrings();
  });
  $("theme").addEventListener("click", () => {
    theme = { auto: "light", light: "dark", dark: "auto" }[theme] || "auto";
    store.set("theme", theme);
    applyTheme();
  });
  window.addEventListener("scroll", () => {
    $("app-bar").classList.toggle("scrolled", window.scrollY > 0);
  });

  // --- media -------------------------------------------------------------------------------

  function addMedia(entries) {
    const known = new Set(state.media.map((m) => m.path));
    for (const entry of entries) {
      if (!known.has(entry.path)) {
        state.media.push({ path: entry.path, kind: entry.kind, name: entry.name || basename(entry.path) });
        known.add(entry.path);
      }
    }
    renderMedia();
  }

  function renderMedia() {
    const list = $("media-list");
    list.replaceChildren();
    for (const item of state.media) {
      const lead =
        item.kind === "dir"
          ? h("span", { class: "lead" }, icon("folder"))
          : h("span", { class: "lead" }, h("img", { src: thumbnail(item.path), alt: "", loading: "lazy" }));
      const supporting = item.kind === "dir" ? t("media.folder_item") : tilde(dirname(item.path));
      list.append(
        h(
          "li",
          { class: "list-item", title: item.path },
          lead,
          h("span", { class: "text" }, h("span", { class: "body-large", text: item.name }), h("span", { class: "body-small muted", text: supporting })),
          iconButton("close", t("media.remove"), () => {
            state.media = state.media.filter((m) => m.path !== item.path);
            renderMedia();
          }),
        ),
      );
    }
    const counts = { image: 0, video: 0, dir: 0 };
    for (const item of state.media) counts[item.kind] += 1;
    const parts = [];
    if (counts.image) parts.push(t("count.photos", { n: counts.image }));
    if (counts.video) parts.push(t("count.videos", { n: counts.video }));
    if (counts.dir) parts.push(t("count.folders", { n: counts.dir }));
    $("media-count").textContent = parts.join(", ");
    $("media-empty").hidden = state.media.length > 0;
    $("media-clear").hidden = state.media.length === 0;
    $("media-ffmpeg").hidden = !(info && !info.ffmpeg && (counts.video || counts.dir));
    renderEnhance();
  }

  $("media-clear").addEventListener("click", () => {
    state.media = [];
    renderMedia();
  });
  $("add-files").addEventListener("click", () => openPicker("files"));
  $("add-folder").addEventListener("click", () => openPicker("folder"));

  // --- settings ----------------------------------------------------------------------------

  function loadSettings() {
    state.settings = { ...DEFAULTS, ...store.get("settings", {}) };
    if (!state.settings.output) state.settings.output = tilde(info.output);
    const available = (s) => info.scales[String(s)];
    if (!available(state.settings.scale) && available(state.settings.scale === 4 ? 8 : 4)) {
      state.settings.scale = state.settings.scale === 4 ? 8 : 4;
    }
  }

  function saveSettings() {
    store.set("settings", state.settings);
  }

  function bindSegmented(id, key, parse = (v) => v) {
    $(id).addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (!button || button.disabled) return;
      state.settings[key] = parse(button.dataset.value);
      saveSettings();
      renderSettings();
    });
  }

  function renderSegmented(id, value) {
    for (const button of $(id).querySelectorAll("button[data-value]")) {
      button.setAttribute("aria-checked", String(button.dataset.value === String(value)));
    }
  }

  function renderSettings() {
    const s = state.settings;
    renderSegmented("scale", s.scale);
    for (const button of $("scale").querySelectorAll("button")) {
      button.disabled = !info.scales[button.dataset.value];
    }
    $("scale-help").textContent = t(`scale.help.${s.scale}`);
    const missing = !info.scales[String(s.scale)];
    $("scale-missing").hidden = !missing;
    $("scale-missing").textContent = t("scale.missing", { scale: s.scale });
    renderSegmented("image-format", s.image_format);
    renderSegmented("video-format", s.video_format);
    $("quality-row").hidden = s.image_format === "png";
    $("quality").value = String(s.quality);
    $("quality-value").textContent = String(s.quality);
    $("quality").style.setProperty("--fill", `${((s.quality - 50) / 50) * 100}%`);
    if ($("output").value !== s.output) $("output").value = s.output;
    $("subfolder").checked = s.subfolder;
    $("overwrite").checked = s.overwrite;
    $("overwrite-row").hidden = s.subfolder;
    $("watermark").checked = s.watermark;
    renderEnhance();
  }

  function renderEnhance() {
    const s = state.settings;
    $("enhance").disabled =
      !info || state.submitting || !state.media.length || !info.scales[String(s.scale)] || !s.output.trim();
  }

  bindSegmented("scale", "scale", Number);
  bindSegmented("image-format", "image_format");
  bindSegmented("video-format", "video_format");
  $("quality").addEventListener("input", () => {
    state.settings.quality = Number($("quality").value);
    saveSettings();
    renderSettings();
  });
  $("output").addEventListener("input", () => {
    state.settings.output = $("output").value;
    saveSettings();
    renderEnhance();
  });
  for (const [id, key] of [
    ["subfolder", "subfolder"],
    ["overwrite", "overwrite"],
    ["watermark", "watermark"],
  ]) {
    $(id).addEventListener("change", () => {
      state.settings[key] = $(id).checked;
      saveSettings();
      renderSettings();
    });
  }
  $("output-browse").addEventListener("click", () => openPicker("output"));

  $("enhance").addEventListener("click", async () => {
    const s = state.settings;
    state.submitting = true;
    renderEnhance();
    try {
      const batch = await api("/api/batches", {
        method: "POST",
        body: {
          inputs: state.media.map((m) => m.path),
          output: s.output.trim(),
          scale: s.scale,
          image_format: s.image_format,
          video_format: s.video_format,
          quality: s.quality,
          watermark: s.watermark,
          overwrite: s.overwrite,
          subfolder: s.subfolder,
        },
      });
      state.batches.set(batch.id, batch);
      renderBatches();
      snackbar(t("enhance.added", { id: batch.id }));
      if (window.matchMedia("(max-width: 920px)").matches) $("results-title").scrollIntoView({ behavior: "smooth" });
      poll();
    } catch (error) {
      fail(error);
    } finally {
      state.submitting = false;
      renderEnhance();
    }
  });

  // --- the file dialog ---------------------------------------------------------------------

  const picker = { mode: "files", path: "", parent: null, entries: [], selected: new Map(), loading: false };

  async function openPicker(mode) {
    picker.mode = mode;
    picker.selected = new Map();
    picker.entries = [];
    picker.path = "";
    const title = { files: "picker.files", folder: "picker.folder", output: "picker.output" }[mode];
    $("picker-title").textContent = t(title);
    const start = mode === "output" ? state.settings.output : store.get("folder", info.home);
    $("picker").showModal();
    await navigate(start, 8); // a folder that is gone (or not made yet): its nearest parent
  }

  async function navigate(path, tries = 1) {
    picker.loading = true;
    renderPicker();
    let target = path;
    for (let left = tries; ; left -= 1) {
      try {
        const data = await api(`/api/folder?${new URLSearchParams({ path: target })}`);
        picker.path = data.path;
        picker.parent = data.parent;
        picker.entries = data.entries;
        if (picker.mode !== "output") store.set("folder", data.path);
        break;
      } catch (error) {
        if (error.status === 404 && left > 1 && dirname(target) !== target) {
          target = dirname(target); // the output folder may not exist yet: show its parent
          continue;
        }
        fail(error);
        if (!picker.path) {
          picker.path = info.home;
          if (target !== info.home) {
            target = info.home;
            continue;
          }
        }
        break;
      }
    }
    picker.loading = false;
    renderPicker();
  }

  function renderPicker() {
    const choosingFiles = picker.mode === "files";
    $("picker-path").value = tilde(picker.path);
    $("picker-up").disabled = !picker.parent;
    const places = $("picker-places");
    places.replaceChildren(
      ...info.places.map((place) =>
        h(
          "button",
          {
            class: "chip",
            type: "button",
            "aria-pressed": String(place.path === picker.path),
            title: place.path,
            onclick: () => navigate(place.path),
          },
          icon(place.id === "home" ? "home" : "folder"),
          t(`place.${place.id}`),
        ),
      ),
    );
    const list = $("picker-list");
    list.replaceChildren();
    if (picker.loading) {
      list.append(h("li", { class: "placeholder center" }, h("div", { class: "spinner" })));
    } else if (!picker.entries.length) {
      list.append(h("li", { class: "placeholder body-medium muted", text: t("picker.empty") }));
    }
    for (const entry of picker.loading ? [] : picker.entries) {
      if (entry.kind === "dir") {
        list.append(
          h(
            "li",
            {
              class: "list-item action",
              role: "button",
              tabindex: 0,
              onclick: () => navigate(entry.path),
              onkeydown: (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  navigate(entry.path);
                }
              },
            },
            h("span", { class: "lead" }, icon("folder")),
            h("span", { class: "text" }, h("span", { class: "body-large", text: entry.name })),
            icon("next"),
          ),
        );
        continue;
      }
      const selected = picker.selected.has(entry.path);
      const toggle = () => {
        if (picker.selected.has(entry.path)) picker.selected.delete(entry.path);
        else picker.selected.set(entry.path, entry);
        renderPicker();
      };
      const lead = h("span", { class: "lead" }, h("img", { src: thumbnail(entry.path), alt: "", loading: "lazy" }));
      const meta = bytes(entry.size);
      list.append(
        h(
          "li",
          choosingFiles
            ? {
                class: "list-item action",
                role: "checkbox",
                tabindex: 0,
                "aria-checked": String(selected),
                onclick: toggle,
                onkeydown: (e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    toggle();
                  }
                },
              }
            : { class: "list-item disabled" },
          choosingFiles ? h("input", { type: "checkbox", tabindex: -1, "aria-hidden": "true", checked: selected }) : null,
          lead,
          h("span", { class: "text" }, h("span", { class: "body-large", text: entry.name }), h("span", { class: "body-small muted", text: meta })),
          entry.kind === "video" ? icon("movie") : null,
        ),
      );
    }
    for (const box of list.querySelectorAll("input[type=checkbox]")) box.checked = box.hasAttribute("checked");
    const files = picker.entries.filter((e) => e.kind !== "dir");
    $("picker-count").hidden = !choosingFiles;
    $("picker-count").textContent = t("picker.selected", { n: picker.selected.size });
    $("picker-all").hidden = !choosingFiles || !files.length;
    $("picker-choose-label").textContent = t(choosingFiles ? "picker.choose" : "picker.choose_folder");
    $("picker-choose").disabled = picker.loading || (choosingFiles ? !picker.selected.size : !picker.path);
  }

  $("picker-up").addEventListener("click", () => picker.parent && navigate(picker.parent));
  $("picker-path").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      navigate($("picker-path").value.trim());
    }
  });
  $("picker-all").addEventListener("click", () => {
    const files = picker.entries.filter((e) => e.kind !== "dir");
    const all = files.every((e) => picker.selected.has(e.path));
    for (const entry of files) {
      if (all) picker.selected.delete(entry.path);
      else picker.selected.set(entry.path, entry);
    }
    renderPicker();
  });
  $("picker-cancel").addEventListener("click", () => $("picker").close());
  $("picker-choose").addEventListener("click", () => {
    if (picker.mode === "files") addMedia([...picker.selected.values()]);
    else if (picker.mode === "folder") addMedia([{ path: picker.path, kind: "dir", name: basename(picker.path) }]);
    else {
      state.settings.output = tilde(picker.path);
      saveSettings();
      renderSettings();
    }
    $("picker").close();
  });

  // --- results -----------------------------------------------------------------------------

  function reasonKind(reason) {
    const text = String(reason || "");
    if (text.startsWith("too large")) return "large";
    if (text.startsWith("too small")) return "small";
    if (text.startsWith("outside")) return "outside";
    return "other";
  }

  function leftAlone(faces) {
    const kinds = {};
    for (const face of faces) if (!face.restored) kinds[reasonKind(face.reason)] = (kinds[reasonKind(face.reason)] || 0) + 1;
    return ["large", "small", "outside", "other"].filter((k) => kinds[k]).map((k) => t(`reason.${k}`, { n: kinds[k] }));
  }

  function faceSummary(item) {
    if (item.kind === "video") {
      const faces = item.faces || {};
      const video = item.video || {};
      return [
        t("item.frames", { n: video.frames ?? 0 }),
        t("item.faces_in", { n: faces.frames_with_faces ?? 0 }),
        t("item.restored", { n: faces.restored ?? 0 }),
      ].join(", ");
    }
    const faces = item.faces || [];
    if (!faces.length) return t("item.no_faces");
    const restored = faces.filter((f) => f.restored).length;
    return [t("item.faces", { n: faces.length }), t("item.restored", { n: restored }), ...leftAlone(faces)].join(", ");
  }

  function statusIcon(status) {
    return { queued: "schedule", loading: "schedule", running: "sparkle", done: "done", failed: "error", cancelled: "cancel" }[status];
  }

  function batchCard(id) {
    const title = h("h3", { class: "title-medium" });
    const chip = h("span", { class: "status" });
    const cancel = h("button", { class: "btn text danger", type: "button" });
    cancel.addEventListener("click", async () => {
      try {
        state.batches.set(id, await api(`/api/batches/${id}/cancel`, { method: "POST" }));
        renderBatches();
      } catch (error) {
        fail(error);
      }
    });
    const progress = h("div", { class: "progress" }, h("span"));
    const counts = h("p", { class: "body-small muted" });
    const message = h("p", { class: "body-medium batch-message" });
    const path = h("span", { class: "body-small path mono" });
    const copy = h("button", { class: "icon-button small", type: "button" }, icon("copy"));
    const open = h("button", { class: "icon-button small", type: "button" }, icon("folder-open"));
    copy.addEventListener("click", () => {
      const folder = state.batches.get(id).folder;
      navigator.clipboard.writeText(folder).then(
        () => snackbar(t("batch.copied")),
        () => snackbar(folder),
      );
    });
    open.addEventListener("click", async () => {
      try {
        await api("/api/open", { method: "POST", body: { path: state.batches.get(id).folder } });
      } catch (error) {
        fail(error);
      }
    });
    const saved = h("span", { class: "label-medium muted" });
    const folder = h("div", { class: "batch-folder" }, saved, path, copy, open);
    const tiles = h("div", { class: "tiles" });
    const el = h(
      "article",
      { class: "card batch" },
      h("div", { class: "card-header" }, title, chip, cancel),
      progress,
      counts,
      message,
      folder,
      tiles,
    );
    return { el, title, chip, cancel, progress, counts, message, path, copy, open, saved, tiles, items: [] };
  }

  function tile(batchId, index) {
    const thumb = h("span", { class: "thumb" });
    const name = h("span", { class: "label-large name" });
    const infoLine = h("span", { class: "body-small info" });
    const bar = h("div", { class: "progress bar" }, h("span"));
    const el = h("button", { class: "tile", type: "button" }, thumb, name, infoLine, bar);
    el.addEventListener("click", () => openViewer(batchId, index));
    return { el, thumb, name, info: infoLine, bar, status: null, image: false };
  }

  function updateTile(view, item) {
    if (!view.image) {
      view.thumb.append(h("img", { src: thumbnail(item.source), alt: "", loading: "lazy" }));
      if (item.kind === "video") view.thumb.append(h("span", { class: "kind" }, icon("movie")));
      view.image = true;
    }
    view.name.textContent = item.name;
    view.el.title = item.source;
    if (view.status !== item.status) {
      for (const old of view.thumb.querySelectorAll(".badge, .overlay")) old.remove();
      if (item.status === "running") view.thumb.append(h("span", { class: "overlay" }, h("div", { class: "spinner" })));
      else if (item.status !== "queued") view.thumb.append(h("span", { class: `badge ${item.status}` }, icon(statusIcon(item.status))));
      view.status = item.status;
    }
    view.el.disabled = item.status !== "done" && item.status !== "failed";
    view.info.classList.toggle("error", item.status === "failed");
    if (item.status === "done") view.info.textContent = faceSummary(item);
    else if (item.status === "failed") view.info.textContent = shorten(item.message) || t("status.failed");
    else if (item.status === "running") view.info.textContent = t("item.running");
    else view.info.textContent = t(`status.${item.status}`);
    const showBar = item.status === "running" && item.kind === "video";
    view.bar.hidden = !showBar;
    if (showBar) view.bar.firstChild.style.setProperty("--value", `${Math.round((item.progress || 0) * 100)}%`);
  }

  function updateCard(card, batch) {
    card.title.textContent = `${t("batch.title", { id: batch.id })} · ×${batch.scale}`;
    card.chip.className = `status ${batch.status}`;
    card.chip.replaceChildren(icon(statusIcon(batch.status)), t(`status.${batch.status}`));
    const active = ACTIVE.includes(batch.status);
    card.cancel.hidden = !active;
    card.cancel.textContent = t("batch.cancel");
    card.progress.hidden = !active;
    card.progress.classList.toggle("indeterminate", batch.status !== "running");
    card.progress.firstChild.style.setProperty("--value", `${Math.round(batch.progress * 100)}%`);
    const c = batch.counts;
    const finished = c.done + c.failed + c.cancelled;
    const parts = [t("batch.of", { done: finished, total: batch.total })];
    if (c.done) parts.push(t("batch.written", { n: c.done }));
    if (c.failed) parts.push(t("batch.failed", { n: c.failed }));
    if (c.cancelled) parts.push(t("batch.cancelled_n", { n: c.cancelled }));
    card.counts.textContent = parts.join(" · ");
    card.message.hidden = !batch.message;
    card.message.textContent = shorten(batch.message);
    card.saved.textContent = t("batch.saved");
    card.path.textContent = tilde(batch.folder);
    card.path.title = batch.folder;
    card.copy.title = t("batch.copy");
    card.copy.setAttribute("aria-label", t("batch.copy"));
    card.open.title = t("batch.open");
    card.open.setAttribute("aria-label", t("batch.open"));
    card.open.hidden = !c.done;
    const items = batch.items || [];
    while (card.items.length < items.length) {
      const view = tile(batch.id, card.items.length);
      card.items.push(view);
      card.tiles.append(view.el);
    }
    items.forEach((item, i) => updateTile(card.items[i], item));
  }

  function renderBatches() {
    const container = $("batches");
    $("results-empty").hidden = state.batches.size > 0;
    const ids = [...state.batches.keys()].sort((a, b) => b - a);
    ids.forEach((id, position) => {
      let card = state.cards.get(id);
      if (!card) {
        card = batchCard(id);
        state.cards.set(id, card);
      }
      updateCard(card, state.batches.get(id));
      if (container.children[position] !== card.el) container.insertBefore(card.el, container.children[position] || null);
    });
    if ($("viewer").open) refreshViewer();
  }

  let pollTimer = null;
  let pollFailures = 0;
  async function poll() {
    clearTimeout(pollTimer);
    const ids = [...state.batches.values()].filter((b) => ACTIVE.includes(b.status)).map((b) => b.id);
    if (!ids.length) return;
    try {
      for (const id of ids) state.batches.set(id, await api(`/api/batches/${id}`));
      pollFailures = 0;
      renderBatches();
    } catch (error) {
      if (error.status === 401) return;
      pollFailures += 1;
      if (pollFailures === 1) fail(error);
    }
    pollTimer = setTimeout(poll, pollFailures ? 3000 : 700);
  }

  // --- the viewer --------------------------------------------------------------------------

  const viewer = { batchId: null, index: 0, tab: "compare", interpolation: store.get("interpolation", "nearest"), source: "target", face: null, key: "" };

  function viewed() {
    const batch = state.batches.get(viewer.batchId);
    return { batch, item: batch?.items?.[viewer.index] };
  }

  function openable(batch) {
    return (batch.items || []).map((item, i) => (item.status === "done" || item.status === "failed" ? i : -1)).filter((i) => i >= 0);
  }

  function openViewer(batchId, index) {
    viewer.batchId = batchId;
    viewer.index = index;
    viewer.face = null;
    viewer.source = "target";
    if (!$("viewer").open) $("viewer").showModal();
    renderViewer();
  }

  function move(step) {
    const { batch } = viewed();
    const order = openable(batch);
    const next = order[order.indexOf(viewer.index) + step];
    if (next !== undefined) {
      viewer.index = next;
      viewer.face = null;
      viewer.source = "target";
      renderViewer();
    }
  }

  function refreshViewer() {
    const { item } = viewed();
    if (item && `${item.status}` !== viewer.key) renderViewer();
  }

  function renderViewer() {
    const { batch, item } = viewed();
    if (!item) return;
    viewer.key = `${item.status}`;
    const order = openable(batch);
    const position = order.indexOf(viewer.index);
    $("viewer-title").textContent = item.name;
    $("viewer-position").textContent = `${t("batch.title", { id: batch.id })} · ${t("viewer.position", { n: position + 1, total: order.length })}`;
    $("viewer-prev").disabled = position <= 0;
    $("viewer-next").disabled = position < 0 || position >= order.length - 1;
    const done = item.status === "done";
    $("viewer-download").hidden = !done;
    if (done) $("viewer-download").href = query("/api/file", { path: item.target, download: "true" });
    if (item.kind === "video" && viewer.tab === "faces") viewer.tab = "compare";
    for (const tab of $("viewer-tabs").querySelectorAll("[role=tab]")) {
      tab.setAttribute("aria-selected", String(tab.dataset.tab === viewer.tab));
      tab.hidden = !done || (tab.dataset.tab === "faces" && item.kind === "video");
    }
    const body = $("viewer-body");
    body.replaceChildren();
    body.scrollTop = 0;
    if (!done) {
      body.append(h("div", { class: "center" }, icon("error"), h("p", { class: "body-large", text: t("viewer.failed") }), h("p", { class: "body-medium mono", text: shorten(item.message) })));
      return;
    }
    if (viewer.tab === "details") body.append(details(batch, item));
    else if (viewer.tab === "faces") body.append(...faces(batch, item));
    else if (item.kind === "video") body.append(...videoView(item));
    else body.append(...compare(item));
  }

  function interpolationToggle() {
    const group = h("div", { class: "segmented", role: "radiogroup", "aria-label": t("viewer.show_as") });
    for (const [value, label] of [
      ["nearest", t("viewer.pixels")],
      ["bicubic", t("viewer.bicubic")],
    ]) {
      group.append(
        h(
          "button",
          {
            type: "button",
            role: "radio",
            "aria-checked": String(viewer.interpolation === value),
            onclick: () => {
              viewer.interpolation = value;
              store.set("interpolation", value);
              renderViewer();
            },
          },
          icon("check", "icon check"),
          label,
        ),
      );
    }
    return h("div", { class: "viewer-toolbar" }, h("span", { class: "label-large", text: t("viewer.show_as") }), group);
  }

  function fit(width, height, longest) {
    const scale = Math.min(1, longest / Math.max(width, height));
    return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
  }

  function compare(item) {
    const out = item.output;
    const preview = fit(out.width, out.height, info.preview_size);
    const body = $("viewer-body");
    const room = { width: Math.max(200, body.clientWidth - 48), height: Math.max(200, window.innerHeight - 230) };
    const scale = Math.min(room.width / out.width, room.height / out.height, preview.width / out.width * 2);
    const shown = { width: Math.round(out.width * scale), height: Math.round(out.height * scale) };
    const after = h("img", { alt: t("viewer.enhanced"), src: query("/api/preview", { path: item.target, ...preview }) });
    const before = h("img", {
      class: "before",
      alt: t("viewer.original"),
      src: query("/api/preview", { path: item.source, ...preview, interpolation: viewer.interpolation }),
    });
    const loading = h("div", { class: "loading" }, h("div", { class: "spinner" }));
    let pending = 2;
    for (const img of [after, before]) {
      img.addEventListener("load", () => {
        pending -= 1;
        if (!pending) loading.remove();
      });
      img.addEventListener("error", () => loading.replaceChildren(icon("error")));
    }
    const knob = h("button", { class: "knob", type: "button", tabindex: -1, "aria-hidden": "true" }, icon("compare"));
    const stage = h(
      "div",
      {
        class: "compare",
        role: "slider",
        tabindex: 0,
        "aria-label": t("viewer.drag"),
        "aria-valuemin": 0,
        "aria-valuemax": 100,
        "aria-valuenow": 50,
      },
      after,
      before,
      h("span", { class: "handle" }),
      knob,
      h("span", { class: "label left label-medium", text: t("viewer.original") }),
      h("span", { class: "label right label-medium", text: t("viewer.enhanced") }),
      loading,
    );
    stage.style.width = `${shown.width}px`;
    stage.style.height = `${shown.height}px`;
    const set = (percent) => {
      const value = Math.min(100, Math.max(0, percent));
      stage.style.setProperty("--pos", `${value}%`);
      stage.setAttribute("aria-valuenow", String(Math.round(value)));
    };
    const fromPointer = (event) => {
      const rect = stage.getBoundingClientRect();
      set(((event.clientX - rect.left) / rect.width) * 100);
    };
    stage.addEventListener("pointerdown", (event) => {
      if (event.target.classList.contains("box")) return;
      stage.setPointerCapture(event.pointerId);
      fromPointer(event);
    });
    stage.addEventListener("pointermove", (event) => {
      if (stage.hasPointerCapture(event.pointerId)) fromPointer(event);
    });
    stage.addEventListener("keydown", (event) => {
      const now = Number(stage.getAttribute("aria-valuenow"));
      const step = event.shiftKey ? 10 : 2;
      if (event.key === "ArrowLeft") set(now - step);
      else if (event.key === "ArrowRight") set(now + step);
      else return;
      event.preventDefault();
      event.stopPropagation();
    });
    (item.faces || []).forEach((face, i) => {
      if (!face.restored || !face.region) return;
      const [x, y, w, h_] = face.region;
      const box = h("span", { class: "box", title: t("viewer.face", { n: i + 1 }) });
      box.style.left = `${(x / out.width) * 100}%`;
      box.style.top = `${(y / out.height) * 100}%`;
      box.style.width = `${(w / out.width) * 100}%`;
      box.style.height = `${(h_ / out.height) * 100}%`;
      box.addEventListener("click", () => {
        viewer.tab = "faces";
        viewer.face = i;
        renderViewer();
      });
      stage.append(box);
    });
    return [interpolationToggle(), h("div", { class: "compare-stage" }, stage)];
  }

  function faces(batch, item) {
    const list = item.faces || [];
    const restored = list.map((face, i) => ({ ...face, n: i + 1, i })).filter((f) => f.restored && f.region);
    const nodes = [interpolationToggle()];
    if (!restored.length) {
      nodes.push(h("p", { class: "body-large center", text: t("viewer.no_faces") }));
    }
    const grid = h("div", { class: "faces" });
    let focused = null;
    for (const face of restored) {
      const [x, y, w, hh] = face.region;
      const pad = 0.15 * Math.max(w, hh);
      const outBox = { x: x - pad, y: y - pad, w: w + 2 * pad, h: hh + 2 * pad };
      const s = batch.scale;
      const inBox = { x: outBox.x / s, y: outBox.y / s, w: outBox.w / s, h: outBox.h / s };
      const size = 320;
      const original = h("img", {
        class: viewer.interpolation === "nearest" ? "pixels" : null,
        alt: t("viewer.original"),
        loading: "lazy",
        src: query("/api/crop", { path: item.source, ...inBox, size, interpolation: viewer.interpolation }),
      });
      const enhanced = h("img", { alt: t("viewer.enhanced"), loading: "lazy", src: query("/api/crop", { path: item.target, ...outBox, size }) });
      const score = face.score === null || face.score === undefined ? "" : ` · ${t("viewer.score", { score: number(face.score, 2) })}`;
      const card = h(
        "article",
        { class: "card face-card" },
        h("h3", { class: "title-small", text: `${t("viewer.face", { n: face.n })}${score}` }),
        h(
          "div",
          { class: "face-pair" },
          h("figure", {}, original, h("figcaption", { class: "label-medium", text: t("viewer.original") })),
          h("figure", {}, enhanced, h("figcaption", { class: "label-medium", text: t("viewer.enhanced") })),
        ),
      );
      if (viewer.face === face.i) focused = card;
      grid.append(card);
    }
    if (restored.length) nodes.push(grid);
    const left = leftAlone(list);
    if (left.length) nodes.push(h("p", { class: "body-medium muted", text: `${t("viewer.left")}: ${left.join(", ")}` }));
    if (focused) {
      requestAnimationFrame(() => {
        focused.scrollIntoView({ block: "center" });
        focused.classList.add("flash");
        setTimeout(() => focused.classList.remove("flash"), 1500);
      });
      viewer.face = null;
    }
    return nodes;
  }

  function videoView(item) {
    const group = h("div", { class: "segmented", role: "radiogroup", "aria-label": t("viewer.compare") });
    const player = h("video", { class: "player", controls: true, preload: "metadata" });
    if (item.output) {
      const body = $("viewer-body");
      const scale = Math.min(1, (body.clientWidth - 48) / item.output.width, (window.innerHeight - 230) / item.output.height);
      player.style.width = `${Math.round(item.output.width * scale)}px`;
      player.style.height = `${Math.round(item.output.height * scale)}px`;
    }
    const message = h("p", { class: "body-medium warning", hidden: true, text: t("viewer.video_unplayable") });
    const load = (keepTime) => {
      const time = keepTime ? player.currentTime : 0;
      const playing = keepTime && !player.paused;
      message.hidden = true;
      player.src = query("/api/file", { path: viewer.source === "target" ? item.target : item.source });
      player.addEventListener(
        "loadedmetadata",
        () => {
          player.currentTime = Math.min(time, player.duration || time);
          if (playing) player.play().catch(() => {});
        },
        { once: true },
      );
    };
    player.addEventListener("error", () => {
      message.hidden = false;
    });
    for (const [value, label] of [
      ["target", t("viewer.enhanced")],
      ["source", t("viewer.original")],
    ]) {
      group.append(
        h(
          "button",
          {
            type: "button",
            role: "radio",
            "aria-checked": String(viewer.source === value),
            onclick: () => {
              viewer.source = value;
              for (const b of group.children) b.setAttribute("aria-checked", String(b === group.children[value === "target" ? 0 : 1]));
              load(true);
            },
          },
          icon("check", "icon check"),
          label,
        ),
      );
    }
    load(false);
    return [h("div", { class: "viewer-toolbar" }, group, h("span", { class: "body-small muted", text: t("viewer.video_hint") })), message, player];
  }

  function details(batch, item) {
    const rows = [];
    const add = (label, value, mono = false) => rows.push(h("dt", { class: "body-medium", text: label }), h("dd", { class: mono ? "mono" : "body-medium", text: value }));
    add(t("details.output"), tilde(item.target), true);
    add(t("details.input"), tilde(item.source), true);
    if (item.input && item.output) {
      add(t("details.size"), t("details.size_value", { iw: item.input.width, ih: item.input.height, ow: item.output.width, oh: item.output.height }));
    }
    if (batch.model) add(t("details.model"), `${batch.model.name} (×${batch.scale})`);
    add(t("details.faces"), faceSummary(item));
    if (item.kind === "video" && item.video) add(t("details.video"), `${t("item.frames", { n: item.video.frames })}, ${t("details.fps", { fps: number(item.video.fps, 2) })}`);
    if (item.seconds !== null && item.seconds !== undefined) add(t("details.time"), t("details.seconds", { s: number(item.seconds, 1) }));
    add(t("details.watermark"), batch.options.watermark ? t("yes") : t("no"));
    if (item.input?.sha256) add(t("details.input_sha"), item.input.sha256, true);
    add(t("details.marks"), t("details.marks_text"));
    return h("dl", { class: "details" }, rows);
  }

  $("viewer-tabs").addEventListener("click", (event) => {
    const tab = event.target.closest("[role=tab]");
    if (!tab) return;
    viewer.tab = tab.dataset.tab;
    renderViewer();
  });
  $("viewer-close").addEventListener("click", () => $("viewer").close());
  $("viewer-prev").addEventListener("click", () => move(-1));
  $("viewer-next").addEventListener("click", () => move(1));
  $("viewer").addEventListener("close", () => {
    for (const player of $("viewer-body").querySelectorAll("video")) player.pause();
    $("viewer-body").replaceChildren();
  });
  $("viewer").addEventListener("keydown", (event) => {
    if (event.target.closest("input, video, .compare")) return;
    if (event.key === "ArrowLeft") move(-1);
    else if (event.key === "ArrowRight") move(1);
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if ($("viewer").open && viewer.tab === "compare" && viewed().item?.kind === "image") renderViewer();
    }, 200);
  });

  // --- about ---------------------------------------------------------------------------------

  $("about").addEventListener("click", () => {
    const text = $("about-text");
    const models = Object.entries(info?.scales || {}).map(([scale, ok]) => `×${scale}: ${t(ok ? "about.available" : "about.missing")}`);
    text.replaceChildren(
      h("p", { text: t("about.text") }),
      h("p", { text: t("about.local") }),
      h("p", { text: t("about.limits") }),
      h(
        "dl",
        {},
        h("dt", { text: t("about.version") }),
        h("dd", { text: info ? `facehall ${info.version}` : "" }),
        h("dt", { text: t("about.models") }),
        h("dd", { text: models.join(", ") }),
        h("dt", { text: t("about.runtime") }),
        h("dd", { text: info ? `${info.backend}, ${info.device}` : "" }),
      ),
    );
    $("about-dialog").showModal();
  });
  $("about-close").addEventListener("click", () => $("about-dialog").close());
  $("snackbar-close").addEventListener("click", () => {
    $("snackbar").hidden = true;
  });

  // --- start ---------------------------------------------------------------------------------

  async function start() {
    try {
      info = await api("/api/info");
    } catch (error) {
      if (error.status !== 401) fail(error);
      return;
    }
    $("locked").hidden = true;
    $("main").hidden = false;
    loadSettings();
    applyStrings();
    try {
      for (const summary of await api("/api/batches")) {
        state.batches.set(summary.id, await api(`/api/batches/${summary.id}`));
      }
    } catch (error) {
      fail(error);
    }
    renderBatches();
    poll();
  }

  async function boot() {
    try {
      const response = await fetch("/static/i18n.json");
      strings = await response.json();
    } catch {
      strings = { en: {} };
    }
    const preferred = (navigator.languages || [navigator.language || "en"]).some((l) => String(l).toLowerCase().startsWith("uk")) ? "uk" : "en";
    lang = store.get("lang", preferred);
    if (!strings[lang]) lang = "en";
    applyStrings();
    readToken();
    if (!token) {
      lock();
      return;
    }
    start();
  }

  // The address from the terminal pasted into an open page changes only the fragment.
  window.addEventListener("hashchange", () => {
    if (/token=/.test(location.hash)) {
      readToken();
      start();
    }
  });

  boot();
})();
