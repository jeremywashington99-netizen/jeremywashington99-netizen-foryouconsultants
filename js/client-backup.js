/* ==========================================================================
   SEGMENT 5 - Client backup & restore (Supabase)

   Export: every row that belongs to a client, from every client table, into
   one JSON file with a SHA-256 checksum.
   Restore: upload that file and every row is upserted back into Supabase,
   in foreign-key order, re-owned by the signed-in account. Running a restore
   twice is safe: rows are matched on their keys, never duplicated.

   Tables covered (client_id links them):
     clients            key id                              (holds the full app view in payload)
     metrics_weekly     key client_id, platform, week_start
     posts              key id
     profile_snapshots  key client_id, platform, captured_on
     documents          key id    (file metadata; the files in Storage are not inside the JSON)
     credentials        key id    (labels and secret references only - no secrets are stored)
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.FYBackup = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const FORMAT = 'fy-client-backup';
  const VERSION = 1;
  const PAGE = 1000;

  const TABLES = [
    { name: 'clients',           match: 'id',        conflict: 'id',                               strip: [],     required: ['id', 'name'] },
    { name: 'metrics_weekly',    match: 'client_id', conflict: 'client_id,platform,week_start',    strip: ['id'], required: ['client_id', 'platform', 'week_start'] },
    { name: 'posts',             match: 'client_id', conflict: 'id',                               strip: [],     required: ['id', 'client_id', 'platform'] },
    { name: 'profile_snapshots', match: 'client_id', conflict: 'client_id,platform,captured_on',   strip: ['id'], required: ['client_id', 'platform', 'captured_on'] },
    { name: 'documents',         match: 'client_id', conflict: 'id',                               strip: [],     required: ['id', 'file_name', 'storage_path'] },
    { name: 'credentials',       match: 'client_id', conflict: 'id',                               strip: [],     required: ['id', 'label'] }
  ];

  /* ---------- checksum ---------- */

  /** JSON with object keys sorted, so the same data always gives the same text. */
  function canonicalJSON(v) {
    if (v === null || typeof v !== 'object') return JSON.stringify(v === undefined ? null : v);
    if (Array.isArray(v)) return '[' + v.map(canonicalJSON).join(',') + ']';
    return '{' + Object.keys(v).sort().filter(function (k) { return v[k] !== undefined; })
      .map(function (k) { return JSON.stringify(k) + ':' + canonicalJSON(v[k]); }).join(',') + '}';
  }

  async function sha256Hex(text) {
    const c = (typeof globalThis !== 'undefined' && globalThis.crypto) || null;
    if (!c || !c.subtle) throw new Error('Web Crypto is not available to compute the checksum');
    const buf = await c.subtle.digest('SHA-256', new TextEncoder().encode(text));
    return Array.from(new Uint8Array(buf)).map(function (b) { return b.toString(16).padStart(2, '0'); }).join('');
  }

  /* ---------- export ---------- */

  async function selectAll(sb, table, col, value) {
    const out = [];
    for (let from = 0; ; from += PAGE) {
      const r = await sb.from(table).select('*').eq(col, value).range(from, from + PAGE - 1);
      if (r.error) throw new Error(table + ': ' + r.error.message);
      const rows = r.data || [];
      Array.prototype.push.apply(out, rows);
      if (rows.length < PAGE) return out;
    }
  }

  /** Read every row for the given client ids from Supabase. */
  async function fetchClientRows(sb, clientIds) {
    const ids = Array.isArray(clientIds) ? clientIds : [clientIds];
    const tables = {};
    TABLES.forEach(function (t) { tables[t.name] = []; });
    for (const id of ids) {
      for (const t of TABLES) Array.prototype.push.apply(tables[t.name], await selectAll(sb, t.name, t.match, id));
    }
    if (!tables.clients.length) throw new Error('No client rows found in the database for: ' + ids.join(', '));
    return tables;
  }

  /** Wrap table rows into a backup object with counts and a checksum. */
  async function buildBackup(tables, meta) {
    const clean = {};
    TABLES.forEach(function (t) { clean[t.name] = (tables[t.name] || []).slice(); });
    const counts = {};
    Object.keys(clean).forEach(function (k) { counts[k] = clean[k].length; });
    return {
      format: FORMAT,
      version: VERSION,
      created_at: (meta && meta.created_at) || new Date().toISOString(),
      source: (meta && meta.source) || 'foryouconsultants.netlify.app',
      client_ids: clean.clients.map(function (c) { return c.id; }),
      client_names: clean.clients.map(function (c) { return c.name; }),
      counts: counts,
      notes: 'Re-upload this file in Settings > Restore from backup. Document files in Storage are not included, only their records.',
      checksum: await sha256Hex(canonicalJSON(clean)),
      tables: clean
    };
  }

  /** One call: read a client (or several) from Supabase and return the backup object. */
  async function exportClients(sb, clientIds, meta) {
    return buildBackup(await fetchClientRows(sb, clientIds), meta);
  }

  function backupFileName(backup, today) {
    const d = today || new Date().toISOString().slice(0, 10);
    const who = backup.client_ids.length === 1 ? String(backup.client_ids[0]).replace(/[^A-Za-z0-9_-]+/g, '-') : 'all-clients';
    return 'fy-backup_' + who + '_' + d + '.json';
  }

  /** Older "Export everything" files were an array of app client objects. Convert them. */
  function fromLegacyClients(list) {
    if (!Array.isArray(list)) throw new Error('not a legacy client list');
    return {
      clients: list.map(function (c) {
        if (!c || !c.id || !c.name) throw new Error('legacy client missing id or name');
        return { id: c.id, name: c.name, industry: c.industry || null, primary_platform: c.primary || null,
                 platforms: c.platforms || [], payload: c, is_active: true };
      })
    };
  }

  /* ---------- verify ---------- */

  async function verifyBackup(obj) {
    const errors = [];
    if (!obj || typeof obj !== 'object') return { ok: false, errors: ['file is not a JSON object'] };
    if (obj.format !== FORMAT) errors.push('not a Four You backup file (format is "' + obj.format + '")');
    if (obj.version !== VERSION) errors.push('unsupported backup version ' + obj.version);
    if (!obj.tables || typeof obj.tables !== 'object') errors.push('backup has no tables');
    if (errors.length) return { ok: false, errors: errors };

    const clientIds = {};
    (obj.tables.clients || []).forEach(function (c) { clientIds[c.id] = true; });
    if (!Object.keys(clientIds).length) errors.push('backup contains no clients');

    TABLES.forEach(function (t) {
      const rows = obj.tables[t.name] || [];
      if (!Array.isArray(rows)) { errors.push(t.name + ' is not a list'); return; }
      rows.forEach(function (r, i) {
        t.required.forEach(function (col) {
          if (r[col] === null || r[col] === undefined || r[col] === '') errors.push(t.name + ' row ' + (i + 1) + ' is missing ' + col);
        });
        if (t.name !== 'clients' && r.client_id && !clientIds[r.client_id]) errors.push(t.name + ' row ' + (i + 1) + ' points at unknown client ' + r.client_id);
      });
    });

    const sum = await sha256Hex(canonicalJSON(normaliseTables(obj.tables)));
    if (sum !== obj.checksum) errors.push('checksum does not match - the file was edited or damaged');
    return { ok: errors.length === 0, errors: errors };
  }

  function normaliseTables(tables) {
    const out = {};
    TABLES.forEach(function (t) { out[t.name] = tables[t.name] || []; });
    return out;
  }

  /* ---------- restore ---------- */

  /**
   * Turn a verified backup into ordered upsert steps.
   * Every row is re-owned by ownerId (Supabase RLS only lets you write your own rows).
   */
  function planRestore(backup, ownerId) {
    if (!ownerId) throw new Error('Not connected to the database - cannot restore without an account');
    return TABLES.map(function (t) {
      const rows = (backup.tables[t.name] || []).map(function (r) {
        const row = Object.assign({}, r);
        t.strip.forEach(function (k) { delete row[k]; });
        row.owner_id = ownerId;
        return row;
      });
      return { table: t.name, onConflict: t.conflict, rows: rows };
    }).filter(function (s) { return s.rows.length; });
  }

  /** Run the steps against Supabase. Returns { table: rowsWritten }. */
  async function runRestore(sb, steps, options) {
    const o = Object.assign({ chunkSize: 500, onProgress: null }, options || {});
    const written = {};
    for (const s of steps) {
      written[s.table] = 0;
      for (let i = 0; i < s.rows.length; i += o.chunkSize) {
        const chunk = s.rows.slice(i, i + o.chunkSize);
        const r = await sb.from(s.table).upsert(chunk, { onConflict: s.onConflict });
        if (r && r.error) {
          const hint = /row-level security/i.test(r.error.message)
            ? ' (a row with the same id belongs to another account)' : '';
          throw new Error('Restore stopped at ' + s.table + ': ' + r.error.message + hint);
        }
        written[s.table] += chunk.length;
        if (o.onProgress) o.onProgress(s.table, written[s.table], s.rows.length);
      }
    }
    return written;
  }

  /** Parse + verify + restore a backup file's text. Accepts legacy client-array files too. */
  async function restoreFromText(sb, text, ownerId, options) {
    let obj;
    try { obj = JSON.parse(text); } catch (e) { throw new Error('File is not valid JSON'); }
    if (Array.isArray(obj)) obj = await buildBackup(fromLegacyClients(obj), { source: 'legacy export' });
    const v = await verifyBackup(obj);
    if (!v.ok) throw new Error('Backup rejected: ' + v.errors.slice(0, 5).join('; '));
    const written = await runRestore(sb, planRestore(obj, ownerId), options);
    return { written: written, client_ids: obj.client_ids, client_names: obj.client_names };
  }

  return { FORMAT: FORMAT, VERSION: VERSION, TABLES: TABLES, canonicalJSON: canonicalJSON, sha256Hex: sha256Hex,
           fetchClientRows: fetchClientRows, buildBackup: buildBackup, exportClients: exportClients,
           backupFileName: backupFileName, fromLegacyClients: fromLegacyClients, verifyBackup: verifyBackup,
           planRestore: planRestore, runRestore: runRestore, restoreFromText: restoreFromText };
});
