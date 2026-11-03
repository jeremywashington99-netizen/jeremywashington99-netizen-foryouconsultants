/* ==========================================================================
   SEGMENT 6 - Site wiring (browser only)

   Connects segments 1-5 to the Four You Consultants page:
     AI Insights  -> Deep Learning Lab card (RNN forecast, CNN images, GAN data)
     Clients      -> a Backup button on every client card
     Settings     -> Back up a client / all clients, Restore from backup

   Uses the page's existing globals: tf, sb, SESSION, DB, PLATFORMS,
   CONTENT_CLASSES, toast, esc, setHTML, download, showModal, closeModal,
   todayISO, logEvent, mkChart, baseOpts, renderClients, renderDashboard.
   ========================================================================== */
(function () {
  'use strict';
  const A = FYDL.activations, R = FYDL.rnn, C = FYDL.cnn, G = FYDL.gan;
  const LAB = { tab: 'rnn', rnn: null, rnnClient: null, cnn: null, cnnImages: [], cnnTask: 'multiclass', gan: null, ganRows: [], ganReal: [] };
  window.FYLab = LAB;

  function el(id) { return document.getElementById(id); }
  function platName(pk) { return (typeof PLATFORMS !== 'undefined' && PLATFORMS[pk] && PLATFORMS[pk].name) || pk; }
  function client() { return DB.active(); }
  function platformsWithWeeks(c, min) {
    return (c && c.platforms || []).filter(function (pk) { const s = c.series[pk]; return s && s.er && s.er.length >= min; });
  }

  /* ---------- audit table (shows the activation policy is met) ---------- */
  function auditHTML(model, task, title) {
    const a = A.auditModel(model, task);
    const rows = a.hidden.map(function (h) {
      return '<tr><td>' + esc(h.name) + '</td><td>' + esc(h.type) + '</td><td>hidden</td><td>' + esc(h.activation === 'leaky' ? 'Leaky ReLU' : h.activation === 'relu' ? 'ReLU' : h.activation) + '</td></tr>';
    }).join('') + '<tr><td>' + esc(a.output.name) + '</td><td>Dense</td><td>output</td><td>' + esc(a.output.activation) + '</td></tr>';
    return '<div class="card-sub" style="margin:12px 0 6px">' + esc(title) + ' &middot; ' +
      (a.ok ? '<span style="color:var(--green)">activation policy met</span>' : '<span style="color:var(--red)">' + esc(a.problems.join('; ')) + '</span>') +
      ' &middot; ' + model.countParams().toLocaleString() + ' parameters</div>' +
      '<div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Layer</th><th>Type</th><th>Role</th><th>Activation</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  /* ---------- lab shell ---------- */
  LAB.render = function () {
    const box = el('dl-lab'); if (!box) return;
    const tabs = [['rnn', 'RNN · weekly forecast'], ['cnn', 'CNN · post images'], ['gan', 'GAN · synthetic weeks']];
    box.innerHTML =
      '<div class="card" style="margin-bottom:14px">' +
      '<div class="card-header"><div><div class="card-title">Deep Learning Lab</div>' +
      '<div class="card-sub">Hidden layers ReLU / Leaky ReLU &middot; output activation matched to each goal &middot; runs in your browser, trains on this client only</div></div></div>' +
      '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">' +
      tabs.map(function (t) {
        return '<button class="btn ' + (LAB.tab === t[0] ? 'btn-primary' : 'btn-ghost') + '" style="font-size:12px;padding:6px 12px" onclick="FYLab.show(\'' + t[0] + '\')">' + t[1] + '</button>';
      }).join('') + '</div><div id="dl-pane"></div></div>';
    ({ rnn: renderRNN, cnn: renderCNN, gan: renderGAN })[LAB.tab]();
  };
  LAB.show = function (t) { LAB.tab = t; LAB.render(); };

  /* ---------- RNN ---------- */
  function renderRNN() {
    const c = client();
    const ready = platformsWithWeeks(c, 4);
    let h = '<div class="callout callout-info" style="margin-bottom:12px">An LSTM reads the last few weeks of each platform (engagement, followers, reach, posts, saves, paid boost) and predicts next week’s engagement rate. Output is <strong>linear</strong> because engagement rate is a continuous number.</div>';
    if (!ready.length) { setHTML('dl-pane', h + '<div class="callout callout-warn">Needs at least 4 weeks on one platform.</div>'); return; }
    h += '<button class="btn btn-primary" id="rnn-btn" onclick="FYLab.trainRNN()">Train RNN</button> <span class="card-sub" id="rnn-status"></span>';
    if (LAB.rnn && LAB.rnnClient === c.id) {
      h += '<div class="grid-2" style="margin-top:14px;margin-bottom:0"><div><div class="card-sub" style="margin-bottom:6px">Training loss</div><div class="chart-wrap" style="height:150px"><canvas id="c-rnn-loss"></canvas></div></div><div id="rnn-forecast"></div></div>';
      h += auditHTML(LAB.rnn.model, 'regression', 'RNN layers');
    }
    setHTML('dl-pane', h);
    if (LAB.rnn && LAB.rnnClient === c.id) {
      mkChart('c-rnn-loss', 'line', { labels: LAB.rnn.history.map(function (_, i) { return i; }),
        datasets: [{ data: LAB.rnn.history, borderColor: '#4FC3F7', pointRadius: 0, borderWidth: 2, tension: .3 }] }, baseOpts(function (v) { return v.toFixed(3); }));
      const rows = LAB.rnn.platforms.map(function (pk) {
        const s = c.series[pk], f = R.forecastRNN(tf, LAB.rnn, s, pk, 4);
        return '<tr><td>' + esc(platName(pk)) + '</td><td>' + s.er[s.er.length - 1] + '%</td>' + f.map(function (v) { return '<td>' + Math.max(0, v) + '%</td>'; }).join('') + '</tr>';
      }).join('');
      setHTML('rnn-forecast', '<div class="card-sub" style="margin-bottom:6px">Engagement forecast (next 4 weeks)</div><div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Platform</th><th>Last</th><th>+1</th><th>+2</th><th>+3</th><th>+4</th></tr></thead><tbody>' + rows + '</tbody></table></div>');
      setText('rnn-status', 'Trained on ' + LAB.rnn.windows + ' windows of ' + LAB.rnn.lookback + ' weeks');
    }
  }
  LAB.trainRNN = async function () {
    const c = client(); const btn = el('rnn-btn');
    const series = {}; platformsWithWeeks(c, 4).forEach(function (pk) { series[pk] = c.series[pk]; });
    if (btn) { btn.disabled = true; btn.textContent = 'Training...'; }
    try {
      if (LAB.rnn) LAB.rnn.model.dispose();
      LAB.rnn = await R.trainRNN(tf, series, { epochs: 150, onEpoch: function (e, n, logs) {
        if (e % 10 === 0) setText('rnn-status', 'Epoch ' + e + ' / ' + n + ' · loss ' + logs.loss.toFixed(4)); } });
      LAB.rnnClient = c.id;
      logEvent('rnn_trained', { client: c.id, windows: LAB.rnn.windows, lookback: LAB.rnn.lookback });
      toast('RNN trained', 'success');
    } catch (e) { LAB.rnn = null; toast('RNN training failed: ' + e.message, 'error', 6000); }
    renderRNN();
  };

  /* ---------- CNN ---------- */
  function renderCNN() {
    const classes = LAB.cnnTask === 'binary' ? ['Low engagement', 'High engagement'] : CONTENT_CLASSES;
    let h = '<div class="callout callout-info" style="margin-bottom:12px">A convolutional network learns from post images. Pick a goal, add images, label each one, then train. ' +
      'Content class uses a <strong>softmax</strong> output (one class out of ' + CONTENT_CLASSES.length + '); high vs low engagement uses a <strong>sigmoid</strong> output (yes / no). Images stay in your browser.</div>' +
      '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px">' +
      '<select id="cnn-task" class="form-input" style="max-width:260px" onchange="FYLab.setCnnTask(this.value)">' +
      '<option value="multiclass"' + (LAB.cnnTask === 'multiclass' ? ' selected' : '') + '>Goal: content class (softmax)</option>' +
      '<option value="binary"' + (LAB.cnnTask === 'binary' ? ' selected' : '') + '>Goal: high vs low engagement (sigmoid)</option></select>' +
      '<label class="btn btn-ghost" style="font-size:12px">+ Add images<input type="file" accept="image/*" multiple style="display:none" onchange="FYLab.addImages(this)"/></label>' +
      '<button class="btn btn-primary" id="cnn-btn" onclick="FYLab.trainCNN()"' + (LAB.cnnImages.length < 4 ? ' disabled' : '') + '>Train CNN</button>' +
      '<span class="card-sub" id="cnn-status">' + LAB.cnnImages.length + ' image(s) &middot; need 4+ across 2+ labels</span></div>';
    h += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px">' + LAB.cnnImages.map(function (im, i) {
      return '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:6px">' +
        '<img src="' + im.url + '" alt="" style="width:100%;height:90px;object-fit:cover;border-radius:6px"/>' +
        '<select class="form-input" style="margin-top:6px;font-size:12px;padding:4px" onchange="FYLab.label(' + i + ',this.value)">' +
        classes.map(function (cl, j) { return '<option value="' + j + '"' + (im.label === j ? ' selected' : '') + '>' + esc(cl) + '</option>'; }).join('') +
        '</select><button class="btn btn-ghost" style="font-size:11px;padding:2px 8px;margin-top:4px;width:100%" onclick="FYLab.removeImage(' + i + ')">Remove</button></div>';
    }).join('') + '</div>';
    if (LAB.cnn) {
      h += '<div style="margin-top:14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap"><label class="btn btn-primary" style="font-size:12px">Predict an image<input type="file" accept="image/*" style="display:none" onchange="FYLab.predictCNN(this)"/></label><span id="cnn-pred" class="card-sub"></span></div>';
      h += auditHTML(LAB.cnn.model, LAB.cnn.task, 'CNN layers');
    }
    setHTML('dl-pane', h);
  }
  LAB.setCnnTask = function (t) {
    LAB.cnnTask = t; LAB.cnnImages.forEach(function (im) { im.label = 0; });
    if (LAB.cnn) { LAB.cnn.model.dispose(); LAB.cnn = null; }
    renderCNN();
  };
  LAB.label = function (i, v) { LAB.cnnImages[i].label = Number(v); };
  LAB.removeImage = function (i) { const im = LAB.cnnImages.splice(i, 1)[0]; if (im) { URL.revokeObjectURL(im.url); im.tensor.dispose(); } renderCNN(); };
  function loadImage(file) {
    return new Promise(function (resolve, reject) {
      const url = URL.createObjectURL(file), img = new Image();
      img.onload = function () {
        const px = tf.browser.fromPixels(img);
        const t = C.preprocessPixels(tf, px, C.DEFAULTS.inputShape[0]); px.dispose();
        resolve({ url: url, tensor: t, name: file.name });
      };
      img.onerror = function () { URL.revokeObjectURL(url); reject(new Error('could not read ' + file.name)); };
      img.src = url;
    });
  }
  LAB.addImages = async function (input) {
    const files = Array.from(input.files || []);
    for (const f of files) {
      try { const im = await loadImage(f); im.label = 0; LAB.cnnImages.push(im); } catch (e) { toast(e.message, 'error'); }
    }
    input.value = ''; renderCNN();
  };
  LAB.trainCNN = async function () {
    const labels = LAB.cnnImages.map(function (im) { return im.label; });
    const distinct = labels.filter(function (v, i, a) { return a.indexOf(v) === i; }).length;
    if (LAB.cnnImages.length < 4 || distinct < 2) { toast('Add at least 4 images with at least 2 different labels', 'error', 5000); return; }
    const btn = el('cnn-btn'); if (btn) { btn.disabled = true; btn.textContent = 'Training...'; }
    const task = LAB.cnnTask, n = task === 'binary' ? 2 : CONTENT_CLASSES.length;
    const xs = tf.stack(LAB.cnnImages.map(function (im) { return im.tensor; }));
    const ys = C.encodeLabels(tf, labels, task, n);
    try {
      if (LAB.cnn) LAB.cnn.model.dispose();
      const model = C.buildCNN(tf, { task: task, nClasses: n });
      await C.trainCNN(tf, model, xs, ys, { epochs: 25, batchSize: 8, onEpoch: function (e, total, logs) {
        setText('cnn-status', 'Epoch ' + (e + 1) + ' / ' + total + ' · loss ' + logs.loss.toFixed(4)); } });
      LAB.cnn = { model: model, task: task };
      logEvent('cnn_trained', { client: (client() || {}).id, images: labels.length, task: task });
      toast('CNN trained on ' + labels.length + ' images', 'success');
    } catch (e) { toast('CNN training failed: ' + e.message, 'error', 6000); }
    finally { xs.dispose(); ys.dispose(); }
    renderCNN();
  };
  LAB.predictCNN = async function (input) {
    const f = input.files && input.files[0]; if (!f || !LAB.cnn) return;
    try {
      const im = await loadImage(f);
      const names = LAB.cnn.task === 'binary' ? null : CONTENT_CLASSES;
      const p = C.predictImage(tf, LAB.cnn.model, im.tensor, names);
      im.tensor.dispose(); URL.revokeObjectURL(im.url);
      setHTML('cnn-pred', LAB.cnn.task === 'binary'
        ? 'High-engagement probability: <strong>' + (p.probability * 100).toFixed(1) + '%</strong>'
        : 'Predicted class: <strong>' + esc(p.label) + '</strong> (' + (p.confidence * 100).toFixed(1) + '%)');
    } catch (e) { toast(e.message, 'error'); }
    input.value = '';
  };

  /* ---------- GAN ---------- */
  function renderGAN() {
    const c = client();
    const ready = platformsWithWeeks(c, 3);
    let h = '<div class="callout callout-info" style="margin-bottom:12px">A generative adversarial network learns what this client’s real weeks look like and generates realistic synthetic weeks for what-if planning. ' +
      'Generator and discriminator hidden layers use <strong>Leaky ReLU</strong>; the generator outputs <strong>sigmoid</strong> (values scaled 0..1), the discriminator outputs <strong>sigmoid</strong> (real vs fake). Synthetic rows are never saved into your real metrics.</div>';
    if (!ready.length) { setHTML('dl-pane', h + '<div class="callout callout-warn">Needs at least 3 weeks on one platform.</div>'); return; }
    h += '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center"><select id="gan-plat" class="form-input" style="max-width:200px">' +
      ready.map(function (pk) { return '<option value="' + pk + '">' + esc(platName(pk)) + '</option>'; }).join('') + '</select>' +
      '<button class="btn btn-primary" id="gan-btn" onclick="FYLab.trainGAN()">Train GAN &amp; generate</button>' +
      (LAB.ganRows.length ? '<button class="btn btn-ghost" onclick="FYLab.downloadGAN()">&#8681; CSV</button>' : '') +
      '<span class="card-sub" id="gan-status"></span></div>';
    if (LAB.ganRows.length) {
      const cols = G.COLUMNS;
      h += '<div class="card-sub" style="margin:14px 0 6px">10 synthetic weeks (' + esc(platName(LAB.ganPlat)) + ') — labelled synthetic</div><div style="overflow-x:auto"><table class="data-table"><thead><tr>' +
        cols.map(function (k) { return '<th>' + k + '</th>'; }).join('') + '</tr></thead><tbody>' +
        LAB.ganRows.map(function (r) { return '<tr>' + cols.map(function (k) { return '<td>' + r[k] + '</td>'; }).join('') + '</tr>'; }).join('') + '</tbody></table></div>';
      const st = G.compareStats(LAB.ganReal, LAB.ganRows);
      h += '<div class="card-sub" style="margin:14px 0 6px">Real vs synthetic averages</div><div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Column</th><th>Real mean</th><th>Synthetic mean</th><th>Real spread</th><th>Synthetic spread</th></tr></thead><tbody>' +
        st.map(function (s) { return '<tr><td>' + s.column + '</td><td>' + s.realMean.toFixed(2) + '</td><td>' + s.synthMean.toFixed(2) + '</td><td>' + s.realStd.toFixed(2) + '</td><td>' + s.synthStd.toFixed(2) + '</td></tr>'; }).join('') + '</tbody></table></div>';
      h += auditHTML(LAB.gan.generator, 'bounded01', 'Generator layers') + auditHTML(LAB.gan.discriminator, 'binary', 'Discriminator layers');
    }
    setHTML('dl-pane', h);
    if (LAB.ganPlat && el('gan-plat')) el('gan-plat').value = LAB.ganPlat;
  }
  LAB.trainGAN = async function () {
    const c = client(), pk = el('gan-plat').value, btn = el('gan-btn');
    const rows = G.seriesToRows(c.series[pk]);
    if (btn) { btn.disabled = true; btn.textContent = 'Training...'; }
    try {
      if (LAB.gan) { LAB.gan.generator.dispose(); LAB.gan.discriminator.dispose(); }
      LAB.gan = await G.trainGAN(tf, rows, { steps: 500, onStep: function (s, n, d, g) {
        setText('gan-status', 'Step ' + s + ' / ' + n + ' · D ' + d.toFixed(3) + ' · G ' + g.toFixed(3)); } });
      LAB.ganRows = G.generate(tf, LAB.gan, 10); LAB.ganReal = rows; LAB.ganPlat = pk;
      logEvent('gan_trained', { client: c.id, platform: pk, real_rows: rows.length });
      toast('Generated 10 synthetic weeks', 'success');
    } catch (e) { toast('GAN training failed: ' + e.message, 'error', 6000); }
    renderGAN();
  };
  LAB.downloadGAN = function () {
    const cols = G.COLUMNS;
    const csv = ['synthetic,platform,' + cols.join(',')].concat(LAB.ganRows.map(function (r) { return 'yes,' + LAB.ganPlat + ',' + cols.map(function (k) { return r[k]; }).join(','); })).join('\n');
    download(new Blob([csv], { type: 'text/csv' }), 'synthetic-weeks_' + (client() || {}).id + '_' + todayISO() + '.csv');
  };

  /* ---------- Backup & restore ---------- */
  function connected() { return !!(sb && SESSION && SESSION.user); }

  async function makeBackup(ids) {
    if (connected()) {
      try { return await FYBackup.exportClients(sb, ids); }
      catch (e) { console.warn('database export failed, using the copy on screen', e); }
    }
    const list = DB.clients.filter(function (c) { return ids.indexOf(c.id) > -1; });
    return FYBackup.buildBackup(FYBackup.fromLegacyClients(list), { source: 'on-screen copy (database not reachable)' });
  }
  async function saveBackup(ids) {
    try {
      const bk = await makeBackup(ids);
      download(new Blob([JSON.stringify(bk, null, 2)], { type: 'application/json' }), FYBackup.backupFileName(bk, todayISO()));
      const n = Object.keys(bk.counts).map(function (k) { return bk.counts[k]; }).reduce(function (a, b) { return a + b; }, 0);
      toast('Backup saved: ' + bk.client_ids.length + ' client(s), ' + n + ' rows', 'success', 4500);
      logEvent('backup_exported', { clients: bk.client_ids, counts: bk.counts });
    } catch (e) { toast('Backup failed: ' + e.message, 'error', 6000); }
  }
  window.fyBackupClient = function (id, ev) { if (ev) ev.stopPropagation(); saveBackup([id]); };
  window.fyBackupAll = function () { saveBackup(DB.clients.map(function (c) { return c.id; })); };
  window.fyChooseBackup = function () {
    showModal('Back up a client',
      '<p style="font-size:13px;color:var(--text2);margin-bottom:14px">Saves every row for the client (profile, weekly metrics, posts, snapshots, document records, credential labels) into one file. Keep it somewhere safe; if the database is ever wiped, upload it under <strong>Restore from backup</strong>.</p>' +
      '<div class="mrow"><label>Client</label><select id="bk-client">' + DB.clients.map(function (c) { return '<option value="' + esc(c.id) + '">' + esc(c.name) + '</option>'; }).join('') + '</select></div>',
      '<button class="btn btn-ghost" onclick="closeModal()">Cancel</button><button class="btn btn-primary" onclick="fyBackupClient(document.getElementById(\'bk-client\').value);closeModal()">&#8681; Save backup</button>');
  };
  window.fyOpenRestore = function () {
    showModal('Restore from backup',
      '<p style="font-size:13px;color:var(--text2);margin-bottom:14px">Upload a <code>fy-backup_...json</code> file. Each row is written back to the database; rows that already exist are updated, never duplicated. The file is checked first and rejected if it was edited or damaged.</p>' +
      '<div class="mrow"><input type="file" id="restore-file" accept="application/json,.json"/></div><div id="restore-status" style="font-size:12px;color:var(--text2);min-height:18px"></div>',
      '<button class="btn btn-ghost" onclick="closeModal()">Close</button><button class="btn btn-primary" onclick="fyRunRestore()">Restore</button>');
  };
  window.fyRunRestore = async function () {
    const f = el('restore-file') && el('restore-file').files[0];
    const st = el('restore-status');
    if (!f) { st.textContent = 'Choose a backup file first.'; return; }
    if (!connected()) { st.textContent = 'Not connected to the database. Reload the page and try again.'; st.style.color = 'var(--red)'; return; }
    st.textContent = 'Checking file...'; st.style.color = 'var(--text2)';
    try {
      const res = await FYBackup.restoreFromText(sb, await f.text(), SESSION.user.id, {
        onProgress: function (t, done, total) { st.textContent = 'Restoring ' + t + ': ' + done + ' / ' + total; } });
      const parts = Object.keys(res.written).map(function (k) { return res.written[k] + ' ' + k; });
      st.style.color = 'var(--green)';
      st.textContent = 'Restored ' + res.client_names.join(', ') + ' — ' + parts.join(', ');
      logEvent('backup_restored', { clients: res.client_ids, written: res.written });
      await DB.load(); renderClients(); await renderDashboard();
      toast('Backup restored', 'success');
    } catch (e) { st.style.color = 'var(--red)'; st.textContent = e.message; }
  };

  /* Re-render the lab whenever AI Insights renders; add Backup buttons to client cards. */
  function wrap(name, after) {
    const orig = window[name]; if (typeof orig !== 'function') return;
    window[name] = function () {
      const r = orig.apply(this, arguments);
      if (r && typeof r.then === 'function') return r.then(function (v) { after(); return v; });
      after(); return r;
    };
  }
  function addBackupButtons() {
    document.querySelectorAll('#clients-grid .client-card').forEach(function (card) {
      if (card.querySelector('.fy-bk')) return;
      const m = /switchClient\('([^']+)'\)/.exec(card.getAttribute('onclick') || '');
      if (!m) return;
      const b = document.createElement('button');
      b.className = 'btn btn-ghost fy-bk'; b.style.cssText = 'font-size:11px;padding:4px 10px;margin-top:10px';
      b.innerHTML = '&#8681; Backup'; b.title = 'Download a restorable backup of this client';
      b.onclick = function (ev) { window.fyBackupClient(m[1], ev); };
      card.appendChild(b);
    });
  }
  wrap('renderAI', LAB.render);
  wrap('renderClients', addBackupButtons);
  document.addEventListener('DOMContentLoaded', function () { setTimeout(function () { LAB.render(); addBackupButtons(); }, 0); });
})();
