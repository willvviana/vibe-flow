/* Editor dialogs and chrome: export/render, send to Create, projects, recovery, shortcut settings, splitters. */
(function () {
  'use strict';
  const App = window.EditorApp, { C, D, S, $, $$, esc } = App;

  // ================================================================== dialog helpers
  App.openDialog = (html, mount) => { const d = $('#dialog'); d.innerHTML = html; $('#modal').hidden = false; if (mount) mount(d); const f = d.querySelector('input,select,button'); if (f) f.focus(); };
  App.closeDialog = () => { $('#modal').hidden = true; $('#dialog').innerHTML = ''; App.capturing = null; };
  $('#modal').addEventListener('mousedown', (e) => { if (e.target.id === 'modal' && !App.exporting) App.closeDialog(); });

  // ================================================================== export / render
  const FORMATS = { wav16: ['WAV 16-bit', 'wav'], wav24: ['WAV 24-bit', 'wav'], wav32: ['WAV 32-bit float', 'wav'], flac: ['FLAC (lossless)', 'flac'], mp3: ['MP3', 'mp3'] };
  const safeName = (s) => String(s || 'mix').replace(/[^\w.\- ()]/g, '_').slice(0, 80) || 'mix';

  App.openExport = (preset) => {
    preset = preset || {};
    const p = App.project, hasSel = App.sel.size > 0, hasRegion = !!App.region;
    if (!p.clips.length) { App.toast('Nothing to export: add audio to the timeline first.'); return; }
    const scope = preset.scope || (hasSel ? 'selected' : hasRegion ? 'region' : 'mix');
    App.openDialog(`<h3>${preset.dest === 'sample' ? 'Send to Create as a sample' : 'Export'}</h3>
      <div class="ed-field"><label for="x-name">File name</label><input type="text" id="x-name" value="${esc(safeName(p.name))}" maxlength="80"></div>
      <div class="ed-field"><label for="x-fmt">Format</label><select id="x-fmt">${Object.entries(FORMATS).map(([k, v]) => `<option value="${k}">${v[0]}</option>`).join('')}</select></div>
      <div class="ed-field" id="x-br-row"><label for="x-br">MP3 quality</label><select id="x-br"><option value="128">128 kbps</option><option value="192" selected>192 kbps</option><option value="256">256 kbps</option><option value="320">320 kbps</option></select></div>
      <div class="ed-field"><label for="x-sr">Sample rate</label><select id="x-sr"><option value="44100">44.1 kHz</option><option value="48000" selected>48 kHz</option><option value="96000">96 kHz</option></select></div>
      <div class="ed-field"><label for="x-scope">What to export</label><select id="x-scope">
        <option value="mix" ${scope === 'mix' ? 'selected' : ''}>Complete mix</option>
        <option value="region" ${scope === 'region' ? 'selected' : ''} ${hasRegion ? '' : 'disabled'}>Selected timeline region${hasRegion ? ` (${C.fmtTime(App.region[0])}–${C.fmtTime(App.region[1])})` : ' (Shift+drag the ruler)'}</option>
        <option value="selected" ${scope === 'selected' ? 'selected' : ''} ${hasSel ? '' : 'disabled'}>Selected clips only</option>
        <option value="tracks">Each track as its own file</option>
        <option value="voice">Voice tracks only</option><option value="music">Music tracks only</option></select></div>
      <div class="ed-field"><label for="x-dest">Destination</label><select id="x-dest">
        <option value="download" ${preset.dest ? '' : 'selected'}>Download to my computer</option><option value="songs">Save to Your Songs</option>
        <option value="sample" ${preset.dest === 'sample' ? 'selected' : ''}>Send to Create as Sample</option></select></div>
      <label class="ed-chk"><input type="checkbox" id="x-lim" ${p.settings.limiter !== false ? 'checked' : ''}> Safety limiter on the master <small>prevents clipping when tracks overlap</small></label>
      <p class="ed-muted" id="x-note"></p>
      <div class="ed-prog" id="x-prog" hidden><i style="width:0"></i></div><div id="x-status" class="ed-muted" role="status"></div>
      <div class="ed-actions"><button class="btn btn-primary" id="x-go">Export</button><button class="ed-btn" id="x-cancel">Close</button></div>`, (d) => {
      const fmt = $('#x-fmt', d), dest = $('#x-dest', d);
      const sync = () => {
        $('#x-br-row', d).hidden = fmt.value !== 'mp3';
        $('#x-note', d).textContent = dest.value === 'songs' ? 'Your Songs stores a WAV of the render (at the chosen sample rate).' : dest.value === 'sample' ? 'Saves a WAV sample, then opens Create with it already loaded.' : '';
        if (dest.value !== 'download' && fmt.value !== 'wav16' && fmt.value !== 'wav24' && fmt.value !== 'wav32') { /* non-WAV is converted to WAV for libraries */ }
      };
      fmt.onchange = dest.onchange = sync; sync();
      $('#x-cancel', d).onclick = () => { if (!App.exporting) App.closeDialog(); };
      $('#x-go', d).onclick = () => runExport({ name: safeName($('#x-name', d).value), fmt: fmt.value, kbps: +$('#x-br', d).value, sr: +$('#x-sr', d).value, scope: $('#x-scope', d).value, dest: dest.value, limiter: $('#x-lim', d).checked }, d);
    });
  };

  const setProg = (d, f, msg) => { const pr = $('#x-prog', d); if (!pr) return; pr.hidden = false; $('i', pr).style.width = Math.round(f * 100) + '%'; $('#x-status', d).textContent = msg || ''; };

  async function encode(chans, sr, o) {
    if (o.fmt.startsWith('wav')) return { blob: new Blob([D.encodeWav(chans, sr, +o.fmt.slice(3) === 32 ? 32 : +o.fmt.slice(3))], { type: 'audio/wav' }), ext: 'wav' };
    if (o.fmt === 'flac') {
      const wav = D.encodeWav(chans, sr, 24);
      const r = await fetch('/api/editor/encode?format=flac&bits=24', { method: 'POST', body: wav });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || 'FLAC encoding failed on the local server.');
      return { blob: new Blob([await r.arrayBuffer()], { type: 'audio/flac' }), ext: 'flac' };
    }
    if (o.fmt === 'mp3') {
      if (!window.lamejs) throw new Error('The MP3 encoder did not load (vendor/lame.min.js). Choose WAV or FLAC.');
      const enc = new window.lamejs.Mp3Encoder(2, sr, o.kbps), n = chans[0].length, parts = [];
      const toI16 = (f) => { const x = new Int16Array(f.length); for (let i = 0; i < f.length; i++) { const v = Math.max(-1, Math.min(1, f[i])); x[i] = v < 0 ? v * 0x8000 : v * 0x7fff; } return x; };
      const L = toI16(chans[0]), R = toI16(chans[1]);
      for (let i = 0, k = 0; i < n; i += 1152, k++) {
        const b = enc.encodeBuffer(L.subarray(i, i + 1152), R.subarray(i, i + 1152));
        if (b.length) parts.push(new Uint8Array(b));
        if (k % 80 === 0) await new Promise((r) => setTimeout(r));          // keep the UI responsive
      }
      const tail = enc.flush(); if (tail.length) parts.push(new Uint8Array(tail));
      return { blob: new Blob(parts, { type: 'audio/mpeg' }), ext: 'mp3' };
    }
    throw new Error('Unknown format.');
  }
  const download = (blob, name) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 20000); };

  /** Which audio is rendered for a scope. Returns [{name, from, to, trackIds, clipIds}] */
  App.planRender = (scope, name) => {
    const p = App.project, end = C.projectDuration(p);
    const span = scope === 'region' && App.region ? App.region : [0, end];
    if (scope === 'selected') {
      const cs = App.clips(); if (!cs.length) throw new Error('Select clips first.');
      return [{ name, from: Math.min(...cs.map((c) => c.start)), to: Math.max(...cs.map((c) => C.clipEnd(c))), trackIds: new Set(p.tracks.map((t) => t.id)), clipIds: new Set(cs.map((c) => c.id)) }];
    }
    if (scope === 'tracks') {
      const jobs = p.tracks.filter((t) => p.clips.some((c) => c.trackId === t.id)).map((t) => ({ name: `${name}-${safeName(t.name)}`, from: span[0], to: span[1], trackIds: new Set([t.id]) }));
      if (!jobs.length) throw new Error('No track has audio.');
      return jobs;
    }
    if (scope === 'voice' || scope === 'music') {
      const ids = p.tracks.filter((t) => (scope === 'voice') === (t.role === 'voice')).map((t) => t.id);
      if (!ids.some((id) => p.clips.some((c) => c.trackId === id))) throw new Error(scope === 'voice' ? 'No audio on a Voice track. Set a track\'s type to Voice in the Track tab.' : 'No audio on a Music track.');
      return [{ name: `${name}-${scope}`, from: span[0], to: span[1], trackIds: new Set(ids) }];
    }
    return [{ name, from: span[0], to: span[1], trackIds: null }];
  };

  async function runExport(o, d) {
    App.exporting = true; $('#x-go', d).disabled = true; $('#x-cancel', d).disabled = true;
    try {
      const jobs = App.planRender(o.scope, o.name);
      setProg(d, 0.02, 'Preparing audio…');
      await App.engine.prepare(App.project, App.assets);
      const hasFx = App.project.clips.some((c) => c.params.reverb.on || c.params.delay.on);
      const results = [];
      for (let i = 0; i < jobs.length; i++) {
        const j = jobs[i], base = i / jobs.length, span = 1 / jobs.length;
        if (j.to - j.from <= 0.01) throw new Error('The selected range is empty.');
        setProg(d, base + span * 0.05, `Rendering ${jobs.length > 1 ? `${i + 1}/${jobs.length} ` : ''}${j.name}…`);
        const r = await App.engine.render(j.from, j.to, o.sr, { trackIds: j.trackIds, clipIds: j.clipIds, limiter: o.limiter, tail: hasFx ? 3 : 0.05, onProgress: (f) => setProg(d, base + span * (0.05 + 0.6 * f), `Rendering ${j.name}…`) });
        setProg(d, base + span * 0.7, 'Encoding…');
        const enc = await encode(r.chans, o.sr, o);
        results.push({ job: j, r, enc });
      }
      const last = results[results.length - 1];
      const st = last.r.stats;
      const summary = `${results.length} file(s) · ${(last.enc.blob.size / 1048576).toFixed(2)} MB · peak ${st.peakDb.toFixed(1)} dBFS · ${st.clipped ? `<b class="ed-err">${st.clipped} clipped samples</b>` : 'no clipping'}`;
      if (o.dest === 'download') {
        for (const x of results) download(x.enc.blob, `${x.job.name}.${x.enc.ext}`);
        setProg(d, 1, ''); $('#x-status', d).innerHTML = `Done: ${summary}`;
      } else {
        const wavBlob = new Blob([D.encodeWav(last.r.chans, o.sr, 16)], { type: 'audio/wav' });
        setProg(d, 0.9, 'Saving…');
        if (o.dest === 'songs') {
          const r = await fetch('/api/editor/save?title=' + encodeURIComponent(last.job.name), { method: 'POST', body: wavBlob });
          const j = await r.json(); if (!r.ok) throw new Error(j.error || 'Could not save to Your Songs.');
          setProg(d, 1, ''); $('#x-status', d).innerHTML = `Saved to Your Songs: ${esc(j.title)} (${summary})`;
        } else {
          const kind = o.scope === 'voice' ? 'vocals' : o.scope === 'music' ? 'instrumental' : 'full';
          const r = await fetch(`/api/samples?name=${encodeURIComponent(last.job.name)}&kind=${kind}`, { method: 'POST', body: wavBlob });
          const j = await r.json(); if (!r.ok) throw new Error(j.error || 'Could not save the sample.');
          await App.saveProject();                                    // keep the editable project so the edit can be reopened later
          setProg(d, 1, ''); $('#x-status', d).textContent = 'Sample saved. Opening Create…';
          setTimeout(() => { App.dirty = false; location.href = 'create.html?sample=' + j.id; }, 500);
        }
      }
      App.lastExport = { stats: st, files: results.map((x) => ({ name: `${x.job.name}.${x.enc.ext}`, size: x.enc.blob.size })) };
    } catch (e) {
      $('#x-status', d).innerHTML = `<span class="ed-err">${esc(e.message)}</span>`; $('#x-prog', d).hidden = true;
    } finally { App.exporting = false; $('#x-go', d).disabled = false; $('#x-cancel', d).disabled = false; }
  }
  App.sendToCreate = () => App.openExport({ dest: 'sample' });

  // ================================================================== projects
  App.openProjects = async () => {
    let list = []; try { list = await S.listProjects(); } catch (e) { /* none */ }
    App.openDialog(`<h3>Projects</h3><div class="ed-actions" style="margin:0 0 8px"><button class="ed-btn" id="pj-new">New project</button></div>
      <div class="ed-list">${list.length ? list.map((p) => `<div class="ed-li" data-id="${p.id}"><div class="nm"><b>${esc(p.name)}</b><br><span class="ed-muted">${new Date(p.modified).toLocaleString()} · ${p.clips} clip(s)</span></div><button class="ed-btn" data-o="open">Open</button><button class="ed-btn" data-o="del">Delete</button></div>`).join('') : '<div class="ed-hint">No saved projects yet. Press Save (Ctrl+S).</div>'}</div>
      <div class="ed-actions"><button class="ed-btn" id="pj-close">Close</button></div>`, (d) => {
      $('#pj-close', d).onclick = App.closeDialog;
      $('#pj-new', d).onclick = async () => { if (App.dirty && !confirm('Discard unsaved changes?')) return; await App.newProject('Untitled project'); App.closeDialog(); };
      d.onclick = async (e) => {
        const b = e.target.closest('[data-o]'); if (!b) return; const id = b.closest('.ed-li').dataset.id;
        if (b.dataset.o === 'del') { if (confirm('Delete this saved project? (Audio files stay in your library.)')) { await S.deleteProject(id); App.openProjects(); } return; }
        if (App.dirty && !confirm('Discard unsaved changes in the current project?')) return;
        const pr = await S.loadProject(id); App.closeDialog(); if (pr) await App.loadProjectData(pr);
      };
    });
  };
  App.offerRecovery = (rec) => {
    App.openDialog(`<h3>Restore your last session?</h3><p class="ed-muted">An unsaved session from ${new Date(rec.saved).toLocaleString()} was found (${rec.project.clips.length} clips). Editor sessions are kept automatically in case the page closed unexpectedly.</p>
      <div class="ed-actions"><button class="btn btn-primary" id="rc-yes">Restore</button><button class="ed-btn" id="rc-no">Discard</button></div>`, (d) => {
      $('#rc-yes', d).onclick = async () => { App.closeDialog(); await App.loadProjectData(rec.project); App.setDirty(true); };
      $('#rc-no', d).onclick = async () => { await S.clearRecovery(); App.closeDialog(); };
    });
  };

  // ================================================================== settings (shortcuts, snapping, crossfade)
  const LABELS = { playPause: 'Play / Pause', undo: 'Undo', redo: 'Redo', copy: 'Copy', paste: 'Paste', cut: 'Cut', selectAll: 'Select all', delete: 'Delete selected clip', duplicate: 'Duplicate clip', save: 'Save project', export: 'Export audio', split: 'Split at playhead', muteTrack: 'Mute selected track', home: 'Playhead to start', end: 'Playhead to end', left: 'Playhead left (Shift = 1 s)', right: 'Playhead right (Shift = 1 s)' };
  App.openSettings = () => {
    const s = App.project.settings;
    const draw = () => App.openDialog(`<h3>Editor settings</h3>
      <h4 style="margin:6px 0">Keyboard shortcuts</h4><p class="ed-muted">Click a shortcut, then press the new key combination. Esc cancels. Ctrl+Mouse wheel zooms the timeline; Shift+click multi-selects.</p>
      <div class="ed-kbd">${Object.entries(LABELS).map(([a, l]) => `<span>${l}</span><button data-a="${a}">${esc(App.shortcuts[a])}</button>`).join('')}</div>
      <h4 style="margin:14px 0 6px">Timeline</h4>
      <div class="ed-field"><label>Snap grid</label><select id="s-grid">${[[0, 'Clip edges and playhead only'], [0.25, '¼ second'], [0.5, '½ second'], [1, '1 second']].map(([v, l]) => `<option value="${v}" ${s.grid === v ? 'selected' : ''}>${l}</option>`).join('')}</select></div>
      <label class="ed-chk"><input type="checkbox" id="s-xf" ${s.crossfade ? 'checked' : ''}> Automatic crossfade where clips on a track overlap</label>
      <label class="ed-chk"><input type="checkbox" id="s-lim" ${s.limiter !== false ? 'checked' : ''}> Safety limiter on the master output (preview and export)</label>
      <div class="ed-actions"><button class="ed-btn" id="s-reset">Reset shortcuts</button><button class="btn btn-primary" id="s-done">Done</button></div>`, (d) => {
      $$('.ed-kbd button', d).forEach((b) => b.onclick = () => {
        b.textContent = 'Press keys…'; b.classList.add('cap'); App.capturing = b.dataset.a;
        const h = (e) => {
          e.preventDefault(); e.stopPropagation();
          if (e.key === 'Escape') { App.capturing = null; document.removeEventListener('keydown', h, true); draw(); return; }
          const combo = C.comboOf(e); if (!combo) return;
          const clash = Object.entries(App.shortcuts).find(([a, c]) => c.toLowerCase() === combo.toLowerCase() && a !== b.dataset.a);
          if (clash) { App.toast(`${combo} is already used for "${LABELS[clash[0]]}".`, true); return; }
          App.shortcuts[b.dataset.a] = combo; try { localStorage.setItem('vf-editor-shortcuts', JSON.stringify(App.shortcuts)); } catch (err) { /* ignore */ }
          App.capturing = null; document.removeEventListener('keydown', h, true); draw();
        };
        document.addEventListener('keydown', h, true);
      });
      $('#s-reset', d).onclick = () => { App.shortcuts = Object.assign({}, C.DEFAULT_SHORTCUTS); try { localStorage.removeItem('vf-editor-shortcuts'); } catch (err) { /* ignore */ } draw(); };
      $('#s-grid', d).onchange = (e) => { s.grid = +e.target.value; App.commit({ quiet: true }); };
      $('#s-xf', d).onchange = (e) => { s.crossfade = e.target.checked; App.commit(); if (App.engine.playing) App.engine.rebuild(); };
      $('#s-lim', d).onchange = (e) => { s.limiter = e.target.checked; App.commit({ quiet: true }); if (App.engine.playing) App.engine.rebuild(); };
      $('#s-done', d).onclick = App.closeDialog;
    });
    draw();
  };

  // ================================================================== chrome: top buttons, splitters, project name
  App.bindChrome = () => {
    $('#btn-play').onclick = () => App.play();
    $('#btn-start').onclick = () => App.seek(0);
    $('#btn-end').onclick = () => App.seek(C.projectDuration(App.project));
    $('#btn-save').onclick = () => App.saveProject();
    $('#btn-projects').onclick = () => App.openProjects();
    $('#btn-settings').onclick = () => App.openSettings();
    $('#btn-export').onclick = () => App.openExport();
    $('#btn-send').onclick = () => App.sendToCreate();
    $('#proj-name').onchange = (e) => { App.project.name = e.target.value.trim().slice(0, 80) || 'Untitled project'; App.commit({ quiet: true }); };
    App.engine.onEnd = () => { App.updatePlay(); App.updatePlayhead(); App.updateTime(); };
    // resizable panels: library width, and the height split between panels and timeline
    const drag = (el, onMove) => {
      el.addEventListener('pointerdown', (e) => { try { el.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } const mv = (ev) => onMove(ev); const up = () => { el.removeEventListener('pointermove', mv); el.removeEventListener('pointerup', up); App.drawAllWaveforms(); App.renderRuler(); }; el.addEventListener('pointermove', mv); el.addEventListener('pointerup', up); e.preventDefault(); });
    };
    drag($('#split-v'), (e) => { const r = $('#ed-mid').getBoundingClientRect(); $('#left').style.width = Math.max(220, Math.min(r.width - 300, e.clientX - r.left - 8)) + 'px'; });
    drag($('#split-h'), (e) => { const r = $('#ed').getBoundingClientRect(); const h = Math.max(120, Math.min(r.height - 260, e.clientY - r.top - $('.ed-top').offsetHeight)); $('#ed-mid').style.height = h + 'px'; });
    for (const [id, axis] of [['split-v', 'x'], ['split-h', 'y']]) {           // keyboard-resizable too
      $('#' + id).addEventListener('keydown', (e) => {
        const d = { ArrowLeft: -20, ArrowUp: -20, ArrowRight: 20, ArrowDown: 20 }[e.key]; if (!d) return;
        if (axis === 'x') $('#left').style.width = Math.max(220, $('#left').offsetWidth + d) + 'px'; else $('#ed-mid').style.height = Math.max(120, $('#ed-mid').offsetHeight + d) + 'px'; e.preventDefault();
      });
    }
    window.addEventListener('dragover', (e) => { if (e.dataTransfer && e.dataTransfer.types.includes('Files')) e.preventDefault(); });
    $('#scroll').addEventListener('drop', (e) => { if (e.dataTransfer.files && e.dataTransfer.files.length) { e.preventDefault(); App.importFiles([...e.dataTransfer.files]); } });
  };

  App.init();
})();
