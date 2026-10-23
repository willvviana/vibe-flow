// Listening Pack page: start a pack (real generations at every adherence level), rate it blind, read the measurements.
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const api = async (u, o = {}) => {
    const r = await fetch(u, { ...o, headers: { 'Content-Type': 'application/json', ...(o.headers || {}) } });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || `Request failed (${r.status})`);
    return d;
  };
  const toast = m => { const t = $('#toast'); t.textContent = m; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => { t.hidden = true; }, 4000); };
  const LV = ['low', 'medium', 'high', 'strict'];
  const cap = s => s[0].toUpperCase() + s.slice(1);
  const num = (v, d = 3) => v == null ? '–' : (+v).toFixed(d);
  const list = s => s.split(',').map(x => x.trim()).filter(Boolean);
  let uploads = [];

  // ---------- sample sources (Sample page recordings, Editor samples, uploads) ----------
  async function loadSources() {
    const box = $('#src-list');
    const [recs, samples] = await Promise.all([api('/api/recordings').catch(() => []), api('/api/samples').catch(() => [])]);
    const ready = recs.filter(r => r.status === 'ready');
    let h = '';
    if (ready.length) {
      h += '<div class="grp-t">Sample page recordings</div>';
      for (const r of ready) for (const [v, lab] of [['full', 'Full'], ['instrumental', 'Instrumental'], ['voice', 'Voice only']].concat(r.vocal_layers && r.vocal_layers.stereo ? [['lead', 'Lead vocal'], ['backing', 'Backing vocals']] : []))
        h += `<label><input type="checkbox" data-type="recording" data-id="${esc(r.id)}" data-version="${v}"> ${esc(r.name || r.id)} · ${lab}</label>`;
    }
    if (samples.length) {
      h += '<div class="grp-t">Editor samples</div>';
      for (const s of samples) h += `<label><input type="checkbox" data-type="sample" data-id="${esc(s.id)}"> ${esc(s.name)} · ${esc(s.kind)} · ${esc(s.duration)} s</label>`;
    }
    if (uploads.length) {
      h += '<div class="grp-t">Uploaded</div>';
      for (const u of uploads) h += `<label><input type="checkbox" checked data-type="upload" data-id="${esc(u.id)}" data-kind="${esc(u.kind)}"> ${esc(u.name)} · ${esc(u.kind)}</label>`;
    }
    box.innerHTML = h || '<p class="muted">No recordings or Editor samples yet. Record one on the Sample page, save one from the Editor, or upload a file.</p>';
    estimate();
  }
  $('#up-file').addEventListener('change', async e => {
    const f = e.target.files[0]; if (!f) return;
    try {
      const r = await api('/api/upload', { method: 'POST', headers: { 'x-filename': encodeURIComponent(f.name), 'Content-Type': 'application/octet-stream' }, body: f });
      uploads.push({ id: r.id, name: r.name, kind: $('#up-kind').value });
      loadSources();
    } catch (err) { toast(err.message); }
    e.target.value = '';
  });

  const chosen = () => $$('#src-list input:checked').map(i => ({ type: i.dataset.type, id: i.dataset.id, version: i.dataset.version, kind: i.dataset.kind }));
  const levels = () => $$('#levels input:checked').map(i => i.value);
  const seeds = () => list($('#seeds').value).map(Number).filter(n => Number.isInteger(n) && n >= 0);
  function estimate() {
    const n = chosen().length * levels().length * seeds().length, s = +$('#secs').value;
    $('#est').textContent = n ? `${n} generation${n > 1 ? 's' : ''} · roughly ${Math.round(n * s * 5.5 / 60)} min on this CPU` : '';
  }
  $('#secs').addEventListener('input', () => { $('#secs-o').textContent = $('#secs').value + ' s'; estimate(); });
  ['#src-list', '#levels', '#seeds'].forEach(s => $(s).addEventListener('change', estimate));
  $('#seeds').addEventListener('input', estimate);

  // ---------- run a pack ----------
  let job = null;
  $('#go').addEventListener('click', async () => {
    const body = { name: $('#pname').value, sources: chosen(), levels: levels(), seeds: seeds(), instruments: list($('#instr').value), genres: list($('#genres').value), seconds: +$('#secs').value };
    if (!body.sources.length) return toast('Choose at least one sample.');
    if (!body.instruments.length && !body.genres.length) return toast('Pick at least one instrument or genre for the new layers.');
    try { job = await api('/api/listening/run', { method: 'POST', body: JSON.stringify(body) }); } catch (e) { return toast(e.message); }
    poll();
  });
  $('#cancel').addEventListener('click', () => { if (job) api(`/api/jobs/${job.id}/cancel`, { method: 'POST' }).catch(() => {}); });
  async function poll() {
    $('#go').disabled = true; $('#cancel').hidden = false; $('#prog').hidden = false;
    while (job) {
      try { job = await api(`/api/jobs/${job.id}`); } catch { break; }
      $('#prog-msg').textContent = job.message || '';
      $('#prog-pct').textContent = `${job.pct || 0} %${job.eta_s ? ` · about ${Math.ceil(job.eta_s / 60)} min left` : ''}`;
      $('#prog-bar').style.width = (job.pct || 0) + '%';
      if (job.status !== 'running') break;
      await new Promise(r => setTimeout(r, 2000));
    }
    $('#go').disabled = false; $('#cancel').hidden = true;
    if (job && job.status === 'complete') { $('#prog').hidden = true; toast('Listening pack ready.'); await loadPacks(); openPack(job.run); }
    else if (job) toast(job.error || job.message || 'Stopped.');
    job = null;
    loadPacks();
  }

  // ---------- packs ----------
  let current = null;
  async function loadPacks() {
    const packs = await api('/api/listening').catch(() => []);
    $('#packs').innerHTML = packs.length ? packs.map(p => `<div class="lp-pk${current === p.id ? ' sel' : ''}"><div><b>${esc(p.name)}</b> <span class="muted">· ${esc(p.created.replace('T', ' '))} · ${esc(p.sources.join(', '))}</span><br>
      <small class="muted">${esc(p.levels.map(cap).join(' / '))} · seeds ${esc(p.seeds.join(', '))} · ${p.status === 'complete' ? (p.failed_checks ? `<span class="tag bad">${p.failed_checks} failed</span>` : '<span class="tag ok">all checks pass</span>') : esc(p.status)}${p.rated ? ' · rated' : ''}</small></div>
      <div class="row" style="margin:0"><button class="btn btn-ghost sm" data-open="${p.id}">Open</button><button class="btn btn-ghost sm" data-del="${p.id}">Delete</button></div></div>`).join('')
      : '<p class="muted">No packs yet.</p>';
  }
  $('#packs').addEventListener('click', async e => {
    const o = e.target.dataset.open, d = e.target.dataset.del;
    if (o) openPack(o);
    if (d && confirm('Delete this listening pack and its audio?')) { try { await api(`/api/listening/${d}`, { method: 'DELETE' }); if (current === d) { current = null; $('#pack').hidden = true; } loadPacks(); } catch (err) { toast(err.message); } }
  });

  const file = (id, p) => `/api/listening/${id}/file?path=${encodeURIComponent(p)}`;
  let rep = null, tab = 'blind';
  async function openPack(id) {
    current = id; rep = await api(`/api/listening/${id}`);
    tab = rep.rating_summary || rep.status !== 'complete' ? 'results' : 'blind';
    renderPack(); loadPacks();
    $('#pack').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  function renderPack() {
    const el = $('#pack'); el.hidden = false;
    el.innerHTML = `<div class="rec-head"><h2>${esc(rep.name)}</h2><span class="muted">${esc(rep.created.replace('T', ' '))}${rep.total_s ? ` · built in ${Math.round(rep.total_s / 60)} min` : ''}</span></div>
      <div class="lp-tabs" role="tablist"><button role="tab" data-tab="blind" aria-selected="${tab === 'blind'}">Blind listening test</button><button role="tab" data-tab="results" aria-selected="${tab === 'results'}">Measured results</button></div>
      <div id="tabbody"></div>`;
    $$('.lp-tabs button', el).forEach(b => b.addEventListener('click', () => { tab = b.dataset.tab; renderPack(); }));
    $('#tabbody').innerHTML = tab === 'blind' ? blindHtml() : resultsHtml();
    if (tab === 'blind') wireBlind();
  }

  // ---------- blind test ----------
  function blindHtml() {
    if (rep.status !== 'complete') return `<p class="muted">The pack is ${esc(rep.status)}${rep.error ? ': ' + esc(rep.error) : ''}.</p>`;
    const bySrc = {};
    for (const b of rep.blind) (bySrc[b.source] = bySrc[b.source] || []).push(b);
    const prev = (rep.rating_summary && rep.rating_summary.rows) || [];
    const old = Object.fromEntries(prev.map(r => [r.code, r]));
    const opt = (v, sel) => `<option value="">–</option>` + [1, 2, 3, 4, 5].map(n => `<option${sel == n ? ' selected' : ''}>${n}</option>`).join('');
    return `<p class="note">Listen to the reference, then to each result in any order. Rate how closely it follows the reference (1 = unrelated, 5 = nearly the same) and how good it sounds (1 = broken, 5 = would use it). Optionally guess its level. The level of each clip stays hidden until you save. Rate the blind test <b>before</b> opening the measured results: they show the levels.</p>
      ${rep.sources.map(s => `<div class="lp-src"><h3>${esc(s.label)} <small class="muted">· ${esc(s.kind)}</small></h3>
        <div class="lp-ref"><small class="muted">Reference</small><audio controls preload="none" src="${file(current, s.ref)}"></audio></div>
        <div class="lp-clips">${(bySrc[s.id] || []).map(b => `<div class="lp-clip" data-code="${b.code}"><div class="h"><b>${b.code}</b>${old[b.code] ? `<span class="muted">was ${esc(cap(old[b.code].level))}, seed ${old[b.code].seed}</span>` : ''}</div>
          <audio controls preload="none" src="${file(current, `blind/${b.code}.wav`)}"></audio>
          <div class="rate"><label>Closeness <select data-k="closeness">${opt(0, old[b.code]?.closeness)}</select></label><label>Quality <select data-k="quality">${opt(0, old[b.code]?.quality)}</select></label>
          <label>Level <select data-k="guess"><option value="">?</option>${LV.map(l => `<option value="${l}"${old[b.code]?.guess === l ? ' selected' : ''}>${cap(l)}</option>`).join('')}</select></label></div></div>`).join('')}</div></div>`).join('')}
      <div class="row"><button class="btn btn-primary" id="save-r">Save ratings and reveal</button></div>
      ${rep.rating_summary ? ratingSummaryHtml(rep.rating_summary) : ''}`;
  }
  function wireBlind() {
    const b = $('#save-r'); if (!b) return;
    b.addEventListener('click', async () => {
      const ratings = {};
      for (const c of $$('.lp-clip')) {
        const g = k => $(`[data-k="${k}"]`, c).value;
        if (g('closeness') || g('quality') || g('guess')) ratings[c.dataset.code] = { closeness: g('closeness'), quality: g('quality'), guess: g('guess') };
      }
      if (!Object.keys(ratings).length) return toast('Rate at least one clip.');
      try { await api(`/api/listening/${current}/ratings`, { method: 'POST', body: JSON.stringify({ ratings }) }); rep = await api(`/api/listening/${current}`); renderPack(); toast('Ratings saved.'); }
      catch (e) { toast(e.message); }
    });
  }
  function ratingSummaryHtml(s) {
    const rho = s.closeness_vs_level_spearman;
    return `<h3 style="margin-top:18px">Your ratings by level</h3><div class="tblw"><table class="lp-t"><thead><tr><th>Level</th><th>Clips rated</th><th>Closeness (1-5)</th><th>Quality (1-5)</th></tr></thead><tbody>
      ${Object.entries(s.per_level).map(([lv, v]) => `<tr><td>${cap(lv)}</td><td>${v.n}</td><td>${num(v.closeness, 2)}</td><td>${num(v.quality, 2)}</td></tr>`).join('')}</tbody></table></div>
      <p class="muted">Rank correlation of closeness with the level: ${rho ? `ρ = ${rho.rho} (p = ${rho.p}, n = ${rho.n})` : 'needs at least 4 rated clips'}.
      Level guessed right: ${s.level_guess_accuracy == null ? '–' : Math.round(s.level_guess_accuracy * 100) + ' %'} (chance ${Math.round(s.chance_guess_accuracy * 100)} %). One listener and a few clips: treat this as a direction, not proof.</p>`;
  }

  // ---------- measured results ----------
  const DIMS = [['adherence_index', 'Adherence index'], ['tempo_sim', 'Tempo (periodicity)'], ['tempo_error_bpm', 'Tempo error (BPM)', 1], ['rhythm_corr', 'Rhythm'], ['chroma_frame_sim', 'Harmony / melody'],
    ['pitch_class_agree', 'Same dominant note'], ['spectral_profile_sim', 'Instrumentation (spectrum)'], ['clap_sim', 'Instrumentation (CLAP)'], ['structure_sim', 'Structure'], ['dynamics_corr', 'Dynamics'], ['ref_presence', 'Reference present']];
  function table(s, levels, part) {
    return `<div class="tblw"><table class="lp-t"><thead><tr><th>${part === 'mix' ? 'Result (mix)' : 'Added layers only'}</th>${levels.map(l => `<th>${cap(l)} <span class="sd">${s[l].adherence}%</span></th>`).join('')}</tr></thead><tbody>
      ${DIMS.map(([k, lab, d]) => `<tr><td>${lab}</td>${levels.map(l => { const v = s[l][part][k] || [null, null]; return `<td>${num(v[0], d ? 1 : 3)}${v[1] ? ` <span class="sd">±${num(v[1], d ? 1 : 3)}</span>` : ''}</td>`; }).join('')}</tr>`).join('')}
      </tbody></table></div>`;
  }
  function bars(s, levels) {
    return `<div class="legend"><span><i style="background:linear-gradient(90deg,#8b5cf6,#22d3ee)"></i>Result (mix)</span><span><i style="background:rgba(244,114,182,.75)"></i>Added layers only</span></div>
      <div class="bars">${levels.map(l => { const a = s[l].mix.adherence_index[0] || 0, b = s[l].layers.adherence_index[0] || 0;
        return `<div class="b"><span>${cap(l)}</span><div class="tr"><i style="width:${Math.max(0, a) * 100}%"></i><u style="left:${Math.max(0, b) * 100}%;width:3px"></u></div><span>${num(a, 2)}</span></div>`; }).join('')}</div>`;
  }
  function vocalHtml(s, levels, src) {
    const rv = src.reference.vocal_layers;
    let h = '';
    if (levels.some(l => s[l].vocals)) {
      h += `<div class="tblw"><table class="lp-t"><thead><tr><th>Vocals in the result vs. the reference</th>${levels.map(l => `<th>${cap(l)}</th>`).join('')}</tr></thead><tbody>
        ${[['pitch_contour_agree', 'Same melody (pitch within 50 cents)'], ['vocal_envelope_corr', 'Vocal phrasing (loudness envelope)'], ['ref_voiced_kept', 'Sung frames kept']].map(([k, lab]) =>
          `<tr><td>${lab}</td>${levels.map(l => { const v = (s[l].vocals || {})[k] || [null]; return `<td>${num(v[0])}${v[1] ? ` <span class="sd">±${num(v[1])}</span>` : ''}</td>`; }).join('')}</tr>`).join('')}</tbody></table></div>`;
    }
    if (rv) {
      const L = rv.lead || {}, B = rv.backing || {};
      h += `<p class="muted"><b>Reference vocal layers.</b> Backing share ${num(rv.backing_share * 100, 1)} % of vocal energy${rv.stereo ? '' : ' (mono vocal: no backing layer can be separated)'}.
        Lead: median pitch ${L.median_f0_hz ?? '–'} Hz, intonation ${L.intonation_cents ?? '–'} cents off the semitone grid, steadiness ${L.pitch_steadiness_cents ?? '–'} cents/10 ms, voiced ${num((L.voiced_ratio || 0) * 100, 0)} % of the time, timing ${rv.lead_timing_ms ?? '–'} ms from the 8th-note grid (chance ~${rv.timing_chance_ms ?? '–'} ms).
        ${rv.backing_share >= 0.05 && rv.stereo ? `Backing: ${B && B.median_f0_hz ? `median pitch ${B.median_f0_hz} Hz, ` : B && B.note ? 'several voices at once, ' : ''}active during ${num((rv.backing_active_while_lead || 0) * 100, 0)} % of the lead.` : ''}
        ${rv.harmony && rv.harmony.intervals && rv.harmony.intervals.length ? `Harmony intervals above the lead: ${rv.harmony.intervals.map(i => esc(i.name)).join(', ')}.` : ''}</p>`;
    }
    return h;
  }
  function instHtml(s, levels, src) {
    const ref = src.reference.instrument_ranks; if (!ref || !Object.keys(ref).length) return '';
    return `<div class="tblw"><table class="lp-t"><thead><tr><th>Selected instrument: CLAP rank (1 = best of ${Object.values(ref)[0].of})</th><th>Reference</th>${levels.map(l => `<th>Layers · ${cap(l)}</th>`).join('')}</tr></thead><tbody>
      ${Object.entries(ref).map(([w, r]) => `<tr><td>${esc(w)} <span class="sd">(${esc(r.label)})</span></td><td>${r.rank}</td>${levels.map(l => { const v = ((s[l].instrument_ranks_layers || {})[w]) || [null]; return `<td>${num(v[0], 1)}</td>`; }).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function resultsHtml() {
    const levels = Object.keys(rep.levels);
    const o = rep.overall;
    let h = rep.status !== 'complete' ? `<p class="alert">Pack ${esc(rep.status)}${rep.error ? ': ' + esc(rep.error) : ''}. Partial results below.</p>` : '';
    if (o) h += `<p class="${o.failed_checks.length ? 'alert' : 'alert info'}">${o.failed_checks.length ? `Failed checks: ${esc(o.failed_checks.join('; '))}.` : 'All automatic checks pass.'} Level ordering is ${o.levels_consistent_across_sources ? 'the same on every sample' : '<b>not</b> the same on every sample'}.</p>`;
    h += `<p class="note">Adherence index = mean of tempo periodicity, rhythm, harmony, instrumentation and structure similarity to the reference (0-1). “Result” is what you get; “added layers only” is the newly generated part on its own, which shows how much the model itself follows the sample, independent of the copied bars. ± = spread across seeds.</p>`;
    for (const src of rep.sources) {
      const s = src.summary; if (!s) { h += `<div class="lp-src"><h3>${esc(src.label)}</h3><p class="muted">${src.runs.length} result(s) so far.</p></div>`; continue; }
      const runsBy = l => src.runs.filter(r => r.level === l);
      h += `<div class="lp-src"><h3>${esc(src.label)} <small class="muted">· ${esc(src.kind)} · ${src.duration_s} s</small></h3>
        <div class="lp-ref"><small class="muted">Reference</small><audio controls preload="none" src="${file(current, src.ref)}"></audio></div>
        ${bars(s, levels)}
        <ul class="checks">${src.checks.map(c => `<li><span class="tag ${c.pass === true ? 'ok' : c.pass === false ? 'bad' : 'info'}">${c.pass === true ? 'PASS' : c.pass === false ? 'FAIL' : 'INFO'}</span><span><b>${esc(c.id)}</b>: ${esc(c.detail)}</span></li>`).join('')}</ul>
        ${table(s, levels, 'mix')}${table(s, levels, 'layers')}${vocalHtml(s, levels, src)}${instHtml(s, levels, src)}
        <div class="lp-clips">${levels.map(l => runsBy(l).map(r => `<div class="lp-clip"><div class="h"><b>${cap(l)} · seed ${r.seed}</b><span class="muted">${r.bars_kept}/${r.bars_total} bars kept</span></div>
          <small class="muted">Result</small><audio controls preload="none" src="${file(current, r.file)}"></audio>
          <small class="muted">Added layers only</small><audio controls preload="none" src="${file(current, r.layers_file)}"></audio>
          <small class="muted">Index ${num(r.mix.adherence_index, 2)} (layers ${num(r.layers.adherence_index, 2)}) · tempo ${r.mix.out_bpm} vs ${r.mix.ref_bpm} BPM · key ${esc(r.mix.out_key)} vs ${esc(r.mix.ref_key)} · click ${r.mix.quality.click_score}</small></div>`).join('')).join('')}</div></div>`;
    }
    if (rep.rating_summary) h += ratingSummaryHtml(rep.rating_summary);
    return h;
  }

  loadSources(); loadPacks();
  const q = new URLSearchParams(location.search).get('pack'); if (q) openPack(q).catch(() => {});
})();
