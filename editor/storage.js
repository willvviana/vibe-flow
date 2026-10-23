/* Persistent storage for the editor (IndexedDB): audio assets (original encoded bytes), saved projects, and a crash-recovery slot.
 * Projects store only references and non-destructive editing instructions; audio bytes live in `assetData`, exports are never stored here. */
(function (root, factory) { root.EditorStorage = factory(); })(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const DB_NAME = 'vibeflow-editor', VERSION = 1;
  let dbp = null;
  function open() {
    if (dbp) return dbp;
    dbp = new Promise((resolve, reject) => {
      const r = indexedDB.open(DB_NAME, VERSION);
      r.onupgradeneeded = () => {
        const db = r.result;
        db.createObjectStore('assetMeta', { keyPath: 'id' });
        db.createObjectStore('assetData', { keyPath: 'id' });
        db.createObjectStore('projects', { keyPath: 'id' });
        db.createObjectStore('recovery', { keyPath: 'key' });
      };
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error || new Error('IndexedDB unavailable (private browsing or storage disabled?)'));
    });
    return dbp;
  }
  const tx = async (stores, mode, fn) => {
    const db = await open();
    return new Promise((resolve, reject) => {
      const t = db.transaction(stores, mode);
      let out;
      Promise.resolve(fn(...[].concat(stores).map((s) => t.objectStore(s)))).then((v) => { out = v; }, reject);
      t.oncomplete = () => resolve(out); t.onerror = () => reject(t.error); t.onabort = () => reject(t.error || new Error('Storage transaction aborted'));
    });
  };
  const req = (r) => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });

  return {
    open,
    /** meta: {id,name,kind('local'|'recording'|'voice'|'song'|'sample'),duration,created,source}; bytes: ArrayBuffer of the encoded file. */
    async putAsset(meta, bytes) { await tx(['assetMeta', 'assetData'], 'readwrite', (m, d) => { m.put(meta); d.put({ id: meta.id, bytes }); }); },
    async getAssetBytes(id) { const r = await tx('assetData', 'readonly', (d) => req(d.get(id))); return r ? r.bytes : null; },
    async listAssets(kinds) { const all = await tx('assetMeta', 'readonly', (m) => req(m.getAll())); return all.filter((a) => !kinds || kinds.includes(a.kind)).sort((a, b) => b.created - a.created); },
    async deleteAsset(id) { await tx(['assetMeta', 'assetData'], 'readwrite', (m, d) => { m.delete(id); d.delete(id); }); },
    async saveProject(p) { await tx('projects', 'readwrite', (s) => { s.put(p); }); },
    async loadProject(id) { return tx('projects', 'readonly', (s) => req(s.get(id))); },
    async listProjects() { const a = await tx('projects', 'readonly', (s) => req(s.getAll())); return a.map((p) => ({ id: p.id, name: p.name, modified: p.modified, clips: (p.clips || []).length })).sort((x, y) => y.modified - x.modified); },
    async deleteProject(id) { await tx('projects', 'readwrite', (s) => { s.delete(id); }); },
    async saveRecovery(p) { await tx('recovery', 'readwrite', (s) => { s.put({ key: 'draft', project: p, saved: Date.now() }); }); },
    async loadRecovery() { const r = await tx('recovery', 'readonly', (s) => req(s.get('draft'))); return r || null; },
    async clearRecovery() { await tx('recovery', 'readwrite', (s) => { s.delete('draft'); }); },
  };
});
