(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  // ---------- API helper ----------
  async function api(url, opt = {}) {
    let r;
    try { r = await fetch(url, opt); } catch { throw new Error('Local engine is offline. Start it with: npm start'); }
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || 'Something went wrong.');
    return d;
  }
  const post = (url, body) => api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });

  // ---------- form ----------
  const GENRES = ['pop', 'rock', 'jazz', 'electronic', 'lo-fi', 'hip hop', 'r&b', 'classical', 'country', 'folk', 'blues', 'funk', 'disco', 'ambient', 'cinematic', 'reggae', 'metal', 'house', 'techno', 'synthwave',
    'soul', 'punk', 'indie', 'latin', 'trap', 'drum and bass', 'dubstep', 'trance', 'gospel', 'afrobeats', 'k-pop', 'reggaeton', 'bossa nova', 'opera', 'new age'];
  const INSTR = ['piano', 'acoustic guitar', 'electric guitar', 'bass', 'drums', 'strings', 'violin', 'cello', 'synthesizer', 'saxophone', 'trumpet', 'flute', 'organ', 'percussion', 'brass', 'harp',
    'electric piano', 'synth bass', 'banjo', 'ukulele', 'accordion', 'clarinet', 'trombone', 'mandolin', 'viola', 'double bass', 'drum machine', 'sitar'];
  const MOODS = ['happy', 'sad', 'energetic', 'calm', 'dark', 'romantic', 'epic', 'mellow', 'uplifting', 'tense', 'dreamy',
    'melancholic', 'aggressive', 'relaxing', 'peaceful', 'groovy', 'nostalgic', 'mysterious', 'playful', 'hopeful', 'intimate', 'angry', 'triumphant', 'sensual', 'eerie', 'festive', 'confident', 'lonely', 'serious'];
  const KEYS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'].flatMap(n => [`${n} major`, `${n} minor`]);
  const prompt = $('#prompt'), count = $('#count');
  // multi-select dropdowns: state lives in `sel`, DOM is re-rendered from it
  const sel = {};
  const LABELS = { 'r&b': 'R&B', 'hip hop': 'Hip-Hop' };
  const fmt = x => LABELS[x] || x.replace(/(^|[\s-])([a-z])/g, (_, p, c) => p + c.toUpperCase());
  const msRender = (id, list) => {
    const root = $(id), chosen = sel[id], open = root.classList.contains('open');
    const label = root.closest('fieldset').querySelector('legend').textContent;
    root.innerHTML =
      `<div class="ms-box" role="combobox" tabindex="0" aria-haspopup="listbox" aria-expanded="${open}" aria-label="${esc(label)}">` +
      `<span class="ms-tags">${chosen.length ? chosen.map(x => `<span class="ms-tag">${esc(fmt(x))}<button type="button" class="ms-x" data-v="${esc(x)}" aria-label="Remove ${esc(fmt(x))}">×</button></span>`).join('') : (root.dataset.placeholder ? `<span class="ms-ph">${esc(root.dataset.placeholder)}</span>` : '')}</span>` +
      `<span class="ms-caret" aria-hidden="true"></span></div>` +
      `<ul class="ms-menu" role="listbox" aria-multiselectable="true">${list.map(x => `<li role="option" tabindex="-1" data-v="${esc(x)}" aria-selected="${chosen.includes(x)}">${esc(fmt(x))}</li>`).join('')}</ul>`;
  };
  const msInit = (id, list, initial) => {
    const root = $(id); sel[id] = [...initial]; msRender(id, list);
    const close = () => { root.classList.remove('open'); $('.ms-box', root).setAttribute('aria-expanded', false); };
    const toggle = v => { const c = sel[id]; sel[id] = c.includes(v) ? c.filter(x => x !== v) : list.filter(x => x === v || c.includes(x)); };
    root.addEventListener('click', e => {
      const x = e.target.closest('.ms-x'), o = e.target.closest('li');
      if (x) { toggle(x.dataset.v); msRender(id, list); return; }
      if (o) { toggle(o.dataset.v); close(); msRender(id, list); $('.ms-box', root).focus(); return; }
      if (e.target.closest('.ms-box')) { const op = root.classList.toggle('open'); $('.ms-box', root).setAttribute('aria-expanded', op); }
    });
    root.addEventListener('keydown', e => {
      const box = $('.ms-box', root), items = $$('li', root), i = items.indexOf(document.activeElement);
      if (e.key === 'Escape') { close(); box.focus(); }
      else if (e.target === box && (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown')) { e.preventDefault(); root.classList.add('open'); box.setAttribute('aria-expanded', true); (items.find(li => li.getAttribute('aria-selected') === 'true') || items[0]).focus(); }
      else if (i >= 0 && e.key === 'ArrowDown') { e.preventDefault(); items[Math.min(i + 1, items.length - 1)].focus(); }
      else if (i >= 0 && e.key === 'ArrowUp') { e.preventDefault(); (items[i - 1] || box).focus(); }
      else if (i >= 0 && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); document.activeElement.click(); }
      else if (e.key === 'Tab') close();
    });
    document.addEventListener('click', e => { if (!root.contains(e.target)) close(); });
    root.setSelected = vals => { sel[id] = list.filter(x => (vals || []).includes(x)); msRender(id, list); };
  };
  msInit('#genres', GENRES, ['r&b']); msInit('#instruments', INSTR, ['piano']); msInit('#moods', MOODS, []);
  $('#key').insertAdjacentHTML('beforeend', KEYS.map(k => `<option>${k}</option>`).join(''));
  $('#tsig').insertAdjacentHTML('beforeend', ['4/4', '3/4', '6/8', '5/4', '7/8'].map(k => `<option>${k}</option>`).join(''));
  const picked = id => [...sel[id]];
  const setPicked = (id, vals) => $(id).setSelected(vals);
  prompt.addEventListener('input', () => { count.textContent = prompt.value.length; });
  $('#duration').addEventListener('input', e => { $('#dur-out').textContent = e.target.value + ' s'; });

  // adherence slider: describes what the chosen percentage actually does (mirrors engine/adherence.py)
  const ADH = [[0, 'New music, inspired by the sample only in general terms. None of the sample is kept.'],
    [10, 'New composition; the sample’s tempo is used. About 3% of its bars are kept (none on short samples).'],
    [25, 'Loosely inspired: tempo and key follow the sample and the new layers continue from its first bar.'],
    [50, 'Balanced: about a third of the sample’s bars are kept as-is; the new layers are aligned to its tempo and key.'],
    [75, 'Close: about two thirds of the bars are kept untouched; layers are aligned and carved around them.'],
    [100, 'Fully close: the whole sample is kept untouched and new layers are added underneath.']];
  const adhText = v => { let d = ADH[0][1]; for (const [k, t] of ADH) if (v >= k) d = t; return d; };
  const adh = $('#adherence'), level = $('#level');
  const adhPaint = () => { $('#adh-out').textContent = adh.value + '%'; $('#adh-desc').textContent = adhText(+adh.value); };
  adh.addEventListener('input', adhPaint); adhPaint();
  level.addEventListener('input', () => { $('#lvl-out').textContent = level.value + ' dB'; });

  let mode = 'text', refId = null;
  $$('.tabs button').forEach(b => b.addEventListener('click', () => {
    mode = b.dataset.mode;
    $$('.tabs button').forEach(x => x.setAttribute('aria-selected', x === b));
    $('#ref-box').hidden = mode === 'text';
    $('#adapt-box').hidden = mode !== 'adapt';
    $('#dur-label').hidden = mode === 'adapt';          // adapt mode works on the sample's own length (max 30 s)
    $('#duration').max = mode === 'melody' ? 30 : 60;
    if (+$('#duration').value > +$('#duration').max) { $('#duration').value = $('#duration').max; $('#dur-out').textContent = $('#duration').max + ' s'; }
    refreshStatus();
  }));

  $('#ref-file').addEventListener('change', async e => {
    const f = e.target.files[0]; if (!f) return;
    const info = $('#ref-info'); info.textContent = 'Checking file…'; refId = null;
    try {
      const r = await api('/api/upload', { method: 'POST', headers: { 'x-filename': encodeURIComponent(f.name), 'Content-Type': 'application/octet-stream' }, body: f });
      refId = r.id;
      info.textContent = `${r.name} · ${r.duration}s · ${r.sample_rate / 1000} kHz · ${r.channels === 1 ? 'mono' : r.channels + ' ch'}`;
    } catch (err) { info.textContent = err.message; info.classList.add('bad'); setTimeout(() => info.classList.remove('bad'), 4000); }
  });

  function collect() {
    return {
      prompt: prompt.value, genres: picked('#genres'), instruments: picked('#instruments'), moods: picked('#moods'),
      bpm: $('#bpm').value, key: $('#key').value, time_signature: $('#tsig').value,
      duration: +$('#duration').value, format: $('#format').value, seed: $('#seed').value, mode, fast: $('#fast').checked,
      reference: mode === 'text' ? null : refId,
      ...(mode === 'adapt' ? { adherence: +adh.value, layer_level_db: +level.value } : {})
    };
  }

  // ---------- engine status & models ----------
  let status = null;
  const need = { text: 'musicgen-small', continue: 'musicgen-small', melody: 'musicgen-melody', adapt: 'musicgen-small' };
  async function refreshStatus() {
    const el = $('#engine-status');
    if (status) paint();                                  // show cached state instantly; refresh below
    try { status = await api('/api/status'); } catch (e) { status = null; el.className = 'engine-status bad'; el.textContent = e.message; $('#models').innerHTML = ''; return; }
    paint();
    renderModels();
  }
  function paint() {
    const el = $('#engine-status');
    const m = status.models.find(x => x.id === need[mode]);
    const hw = status.hardware;
    const deps = status.dependencies && !status.dependencies.ok ? ` <b>Dependency problem:</b> ${esc(status.dependencies.problems[0])}` : '';
    el.className = 'engine-status ' + (m.installed && !deps ? 'ok' : 'warn');
    el.innerHTML = m.installed
      ? `Local engine ready · ${esc(m.name)} on ${hw.device.toUpperCase()}${hw.device === 'cpu' ? ' (slow)' : ''}${deps}`
      : m.integrity === 'damaged'
        ? `${esc(m.name)} files look damaged. <button type="button" class="link" data-act2="repair" data-model="${m.id}">Repair…</button>`
        : `${esc(m.name)} isn't installed (${m.download_gb} GB download). <button type="button" class="link" data-install="${m.id}">Install…</button>`;
  }
  function renderModels() {
    const hw = status.hardware;
    $('#models').innerHTML = `<p class="muted">${esc(hw.os)} · ${hw.cpu_threads} threads · ${hw.ram_available_gb}/${hw.ram_total_gb} GB RAM free · GPU: ${esc(hw.gpus.join(', ') || 'none')} · PyTorch ${esc(hw.torch || 'missing')} · ${esc(hw.device_note)}</p>` +
      status.models.map(m => `<div class="mrow"><div><b>${esc(m.name)}</b> <span class="muted">${m.params_m}M params · ~${m.ram_gb} GB RAM · ${esc(m.license)}${m.commercial_ok ? '' : ' · non-commercial'}</span><br><span class="muted">${esc(m.notes)}</span></div>
        ${m.installed ? `<span class="tag-ok">Installed${m.integrity === 'ok' ? ' · verified' : ' · unverified'}</span> <button type="button" class="link" data-act2="verify" data-model="${m.id}">Verify</button> <button type="button" class="link" data-act2="repair" data-model="${m.id}">Repair</button>` : m.integrity === 'damaged' ? `<button type="button" class="btn btn-ghost" data-act2="repair" data-model="${m.id}">Repair</button>` : `<button type="button" class="btn btn-ghost" data-install="${m.id}">Install (${m.download_gb} GB)</button>`}</div>`).join('') +
      `<p class="muted"><b>Not enabled:</b></p>` + status.unsupported.map(u => `<p class="muted"><b>${esc(u.name)}</b>: ${esc(u.reason)}</p>`).join('');
  }
  document.addEventListener('click', async e => {
    const b = e.target.closest('[data-install]'); if (!b) return;
    const m = status.models.find(x => x.id === b.dataset.install);
    if (!confirm(`Download ${m.name}?\n\nSize: about ${m.download_gb} GB from Hugging Face (${m.repo}).\nLicence: ${m.license}.`)) return;
    try { await runJob(await post('/api/models/install', { model: m.id }), 'Downloading model'); await refreshStatus(); } catch (err) { show(esc(err.message), 'err'); }
  });

  document.addEventListener('click', async e => {
    const b = e.target.closest('[data-act2]'); if (!b) return;
    const m = status.models.find(x => x.id === b.dataset.model);
    if (b.dataset.act2 === 'repair' && !confirm(`Re-download ${m.name} (about ${m.download_gb} GB)? Existing files will be replaced.`)) return;
    try { await runJob(await post('/api/models/' + b.dataset.act2, { model: m.id }), b.dataset.act2 === 'repair' ? 'Repairing model' : 'Verifying model'); show(b.dataset.act2 === 'verify' ? 'Model files verified against the published checksum.' : 'Model repaired and verified.'); await refreshStatus(); }
    catch (err) { show(esc(err.message), 'err'); await refreshStatus(); }
  });

  // ---------- jobs ----------
  const go = $('#go'), cancel = $('#cancel'), out = $('#result');
  const show = (html, cls = '') => { out.innerHTML = `<div class="res ${cls}">${html}</div>`; };
  let currentJob = null, lastSettings = null;

  async function runJob(job, title) {
    currentJob = job.id; go.disabled = true; cancel.hidden = false;
    try {
      while (job.status === 'running') {
        out.innerHTML = `<div class="res"><div class="bar"><i style="width:${job.pct}%"></i></div><div class="pm"><span><span class="spin"></span>${esc(title)}: ${esc(job.stage)}</span><span>${job.pct}%</span></div><small>${esc(job.message)}</small></div>`;
        await sleep(1000);
        job = await api('/api/jobs/' + job.id);
      }
      if (job.status === 'cancelled') { show('Cancelled.'); return null; }
      if (job.status === 'failed') throw new Error(job.error);
      return job;
    } finally { go.disabled = false; cancel.hidden = true; currentJob = null; }
  }
  cancel.addEventListener('click', () => { if (currentJob) post(`/api/jobs/${currentJob}/cancel`).catch(() => {}); });

  const fmtDur = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
  function adaptDetails(t) {
    const a = t.analysis, p = t.plan, e = t.evaluation;
    return `<div class="cmp"><div><small>Original sample (analysed window)</small><audio controls preload="none" src="/api/library/${t.id}/audio?part=source"></audio></div>
      <div><small>Added layers only</small><audio controls preload="none" src="/api/library/${t.id}/audio?part=layers"></audio></div></div>
      <p class="muted">Adherence ${esc(p.adherence)}% → ${p.retained_bars.filter(Boolean).length} of ${p.retained_bars.length} bars kept as the original, ${p.context_bars} bar(s) of audio conditioning, tempo/key hints ${p.tempo_hint >= 0.5 ? 'on' : 'off'}/${p.key_hint >= 0.5 ? 'on' : 'off'}, alignment ${Math.round(p.align_strength * 100)}%.<br>
      Sample: ${esc(a.bpm)} BPM, ${esc(a.key.name)}, ${a.n_bars} bars, chords ${esc((a.chords_per_bar || []).slice(0, 8).join(' '))}${a.vocals_probability != null ? `, singing/speech probability ${esc(a.vocals_probability)}` : ''}.<br>
      Layers: tempo error ${esc(e.layer_tempo_error_bpm)} BPM, key ${esc(e.layer_key)}, rhythmic overlap with the sample ${esc(e.rhythmic_duplication_onset_corr)}; sample integrity residual ${esc(e.source_integrity_max_residual)}.</p>`;
  }
  function trackCard(t, full) {
    const a = t.audio, f = Object.keys(a.files), c = t.controls || {};
    const base = `/api/library/${t.id}/audio`;
    return `<div class="clip" data-id="${t.id}">
      <div class="clip-h"><b class="ttl">${esc(t.title)}</b><span>${fmtDur(a.duration)} · ${a.sample_rate / 1000} kHz</span></div>
      <audio controls preload="none" src="${base}?format=wav"></audio>${t.kind === 'adapt' && full ? adaptDetails(t) : ''}
      <div class="dl"><a href="${base}?format=wav&download=1">WAV</a><a href="${base}?format=flac&download=1">FLAC</a><a href="${base}?format=mp3&download=1">MP3</a><a href="${base}?format=wav&normalize=1&download=1">WAV (normalised)</a>
        <button type="button" class="link" data-act="rename">Rename</button><button type="button" class="link" data-act="regen">Regenerate</button><button type="button" class="link danger" data-act="delete">Delete</button></div>
      ${full ? `<details open><summary>Details</summary><p class="muted">Model: ${esc(t.model.name)} on ${esc(t.device)}${t.fast_mode ? ' (fast int8)' : ''} · seed ${t.seed}<br>
        Prompt sent to the model: <em>${esc(t.model_prompt)}</em><br>
        Exact: ${esc((c.enforced || []).join(', '))}<br>Approximate: ${esc((c.approximate || []).join(', ') || 'none')}<br>
        Loaded in ${t.timing.load_s}s, generated in ${t.timing.generate_s}s${t.timing.real_time_factor ? ` (${t.timing.real_time_factor}× real time)` : ''}${t.timing.peak_memory_gb ? `, peak memory ${t.timing.peak_memory_gb} GB` : ''} · peak ${a.analysis.peak_dbfs} dBFS${t.reference ? `<br>Reference: ${esc(t.reference.name)} (${esc(t.reference.workflow)})` : ''}</p>
        ${(t.warnings || []).map(w => `<p class="warn-line">⚠ ${esc(w)}</p>`).join('')}</details>` : ''}</div>`;
  }

  async function generate(settings) {
    lastSettings = settings;
    try {
      const job = await runJob(await post('/api/generate', { settings }), 'Generating');
      if (!job) return;
      out.innerHTML = `<div class="res">${trackCard(job.track, true)}</div>`;
      loadLibrary();
    } catch (e) {
      show(`${esc(e.message)} <button type="button" class="link" id="retry">Try again</button>`, 'err');
    }
  }
  go.addEventListener('click', () => {
    if (!prompt.value.trim() && !picked('#genres').length && !picked('#instruments').length) { show('Describe your music or pick a genre or instrument first.', 'err'); return; }
    if (mode !== 'text' && !refId) { show('Choose a reference audio file first.', 'err'); return; }
    generate(collect());
  });
  out.addEventListener('click', e => { if (e.target.id === 'retry' && lastSettings) generate(lastSettings); });

  // ---------- library ----------
  let tracks = [];
  async function loadLibrary() {
    try { tracks = await api('/api/library'); } catch { return; }
    $('#lib-count').textContent = tracks.length ? `(${tracks.length})` : '';
    $('#library').innerHTML = tracks.length ? tracks.map(t => trackCard(t, false)).join('') : '<p class="muted">Nothing here yet. Generated tracks are saved locally in the library folder.</p>';
  }
  async function trackAction(e) {
    const b = e.target.closest('[data-act]'); if (!b) return;
    const id = b.closest('.clip').dataset.id, t = tracks.find(x => x.id === id) || {};
    try {
      if (b.dataset.act === 'delete') { if (!confirm('Delete this track and its exports?')) return; await api('/api/library/' + id, { method: 'DELETE' }); loadLibrary(); out.innerHTML = ''; }
      if (b.dataset.act === 'rename') { const n = prompt_('New title', t.title); if (n) { await api('/api/library/' + id, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: n }) }); loadLibrary(); } }
      if (b.dataset.act === 'regen') {
        const s = t.original_settings; if (!s) return;
        if (s.reference) { show('Regenerating from a reference needs the file again. Upload it, then press Generate.', 'err'); return; }
        const { reference, seed, ...rest } = s;
        $('#studio').scrollIntoView({ behavior: 'smooth' }); generate({ ...rest, seed: '' });
      }
    } catch (err) { show(esc(err.message), 'err'); }
  }
  const prompt_ = (m, d) => window.prompt(m, d);
  $('#library').addEventListener('click', trackAction);
  out.addEventListener('click', trackAction);

  refreshStatus(); loadLibrary();

  // ---------- saved samples (from the Editor: "Send to Create as Sample") ----------
  async function useSample(id) {
    const r = await api(`/api/samples/${id}/use`, { method: 'POST' });
    refId = r.id;
    const info = $('#ref-info');
    info.textContent = `${r.name} · ${r.duration}s · ${r.sample_rate / 1000} kHz · ${r.channels === 1 ? 'mono' : r.channels + ' ch'} · from the Editor`;
    return r;
  }
  async function fillSamplePicker() {
    const pick = $('#sample-pick'); if (!pick) return;
    try {
      const list = await api('/api/samples');
      pick.innerHTML = '<option value="">Use a saved sample…</option>' + list.map(s => `<option value="${esc(s.id)}">${esc(s.name)} (${esc(s.kind)}, ${Math.round(s.duration)}s)</option>`).join('');
      pick.hidden = !list.length;
    } catch { pick.hidden = true; }
  }
  if ($('#sample-pick')) $('#sample-pick').addEventListener('change', async e => { if (e.target.value) { try { await useSample(e.target.value); } catch (err) { show(esc(err.message), 'err'); } } });
  fillSamplePicker();
  (async () => {                                                  // opened from the Editor with ?sample=<id>: load it as the sample input
    const m = /[?&]sample=([a-f0-9]{16})/.exec(location.search); if (!m) return;
    try {
      await useSample(m[1]);
      const tab = $$('.tabs button').find(b => b.dataset.mode === 'adapt'); if (tab) tab.click();
      history.replaceState(null, '', location.pathname);
    } catch (err) { show(esc(err.message), 'err'); }
  })();

  // ---------- recordings from the Sample page: choose Full / Instrumental / Voice, fields filled from the audio analysis ----------
  const ALIAS = {
    genres: { 'heavy metal': 'metal', 'electronic dance music': 'electronic', 'orchestral film score': 'cinematic', 'singer-songwriter acoustic': 'folk', 'punk rock': 'punk', 'alternative rock': 'rock', 'indie rock': 'indie', 'chillout': 'ambient', 'hip hop': 'hip hop', 'rap': 'hip hop', "children's music": 'pop' },
    instruments: { 'drum kit': 'drums', 'bass guitar': 'bass', 'distorted electric guitar': 'electric guitar', 'synthesizer pads': 'synthesizer', 'synthesizer lead': 'synthesizer', 'string section': 'strings', 'brass section': 'brass', 'hand drums': 'percussion', 'bells and mallets': 'percussion' },
    moods: {},
  };
  const pickTags = (list, key, allowed, min, max) => {
    const out = [];
    for (const t of list || []) {
      if (typeof t.score === 'number' && t.score < min && !t.edited) continue;
      const l = String(t.label || '').toLowerCase(), v = ALIAS[key][l] || l;
      if (allowed.includes(v) && !out.includes(v)) out.push(v);
      if (out.length >= max) break;
    }
    return out;
  };
  function fillFromAnalysis(an) {
    const t = an && an.tags; if (!t) return [];
    const filled = [];
    const g = pickTags(t.genres, 'genres', GENRES, 0.08, 3), i = pickTags(t.instruments, 'instruments', INSTR, 0.06, 5), m = pickTags(t.moods, 'moods', MOODS, 0.08, 2);
    if (g.length) { setPicked('#genres', g); filled.push('genre'); }
    if (i.length) { setPicked('#instruments', i); filled.push('instruments'); }
    if (m.length) { setPicked('#moods', m); filled.push('mood'); }
    if (t.bpm && t.bpm.value >= 40 && t.bpm.value <= 220) { $('#bpm').value = Math.round(t.bpm.value); filled.push('BPM'); }
    if (t.key && KEYS.includes(t.key.value)) { $('#key').value = t.key.value; filled.push('key'); }
    return filled;
  }
  const VERSION_LABEL = { full: 'Full recording', instrumental: 'Instrumental track', voice: 'Voice only' };
  let recCur = null;
  async function useRecording(id, version, fill) {
    const r = await api(`/api/recordings/${id}/use?version=${version}`, { method: 'POST' });
    refId = r.id; recCur = { id, version };
    $('#ref-info').textContent = `${r.name} · ${r.duration}s · from the Sample page`;
    $('#rec-box').hidden = false;
    const radio = $(`input[name=recver][value=${version}]`); if (radio) radio.checked = true;
    const filled = fill ? fillFromAnalysis(r.analysis) : [];
    $('#rec-note').textContent = fill
      ? (filled.length ? `Filled ${filled.join(', ')} from the audio analysis. Change anything before generating.` : 'No analysis tags were available for this recording; set the fields yourself.')
      : `Now using: ${VERSION_LABEL[version]}. Your genre, instrument, mood and tempo choices were kept.`;
    if (version === 'voice') $('#rec-note').textContent += ' The voice stem guides rhythm, key and tempo only; the music generated is instrumental.';
    return r;
  }
  $$('input[name=recver]').forEach(rd => rd.addEventListener('change', async () => {
    if (!recCur || !rd.checked) return;
    try { await useRecording(recCur.id, rd.value, false); } catch (err) { show(esc(err.message), 'err'); }
  }));
  async function fillRecordingPicker() {
    const pick = $('#rec-pick'); if (!pick) return;
    try {
      const list = (await api('/api/recordings')).filter(x => x.status === 'ready');
      pick.innerHTML = '<option value="">Use a recording…</option>' + list.map(x => `<option value="${esc(x.id)}">${esc(x.name)} (${Math.round(x.duration)}s)</option>`).join('');
      pick.hidden = !list.length;
    } catch { pick.hidden = true; }
  }
  if ($('#rec-pick')) $('#rec-pick').addEventListener('change', async e => { if (e.target.value) { try { await useRecording(e.target.value, 'full', true); } catch (err) { show(esc(err.message), 'err'); } } });
  fillRecordingPicker();
  (async () => {                                                  // opened from the Sample page with ?rec=<id>&v=<version>
    const m = /[?&]rec=([a-f0-9]{16})/.exec(location.search); if (!m) return;
    const v = (/[?&]v=(full|instrumental|voice)/.exec(location.search) || [])[1] || 'full';
    try {
      await useRecording(m[1], v, true);
      const tab = $$('.tabs button').find(b => b.dataset.mode === 'adapt'); if (tab) tab.click();
      history.replaceState(null, '', location.pathname);
    } catch (err) { show(esc(err.message), 'err'); }
  })();

  // preset chosen from the homepage showcase
  try {
    const p = JSON.parse(sessionStorage.getItem('vf-preset') || 'null'); sessionStorage.removeItem('vf-preset');
    if (p) {
      prompt.value = p.text; count.textContent = p.text.length;
      setPicked('#genres', p.genres); setPicked('#instruments', p.instruments); setPicked('#moods', p.moods);
      $('#bpm').value = p.bpm;
    }
  } catch {}
})();
