const $ = id => document.getElementById(id);
const esc = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = value => value == null ? '—' : value === 0 ? '0' : Number(value).toExponential(2);
const statuses = {pass:'Within contract',divergence:'Divergence',invariant_violation:'Contract issue',execution_error:'Execution error',unsupported:'Unsupported'};
const icons = {softmax:'σ',masking:'⊞',normalization:'μ',metrics:'∠',quantization:'↳'};
const severity = {divergence:0,invariant_violation:1,execution_error:2,unsupported:3,pass:4};
let report, page = 0, view = 'reports', selectedStatus = '', selectedObservation = null;
const PAGE_SIZE = 12;
const safePath = path => typeof path === 'string' && /^(artifacts|observations)\/[a-f0-9]{64}\/[a-z0-9_.-]+$/.test(path) ? path : '#';
const badge = status => `<span class="badge ${esc(status)}"><span>${status === 'pass' ? '✓' : status === 'divergence' ? '↗' : '·'}</span>${esc(statuses[status] || status)}</span>`;
function setURL() {
  const params = new URLSearchParams({view});
  for (const name of ['search','category','dtype','optimization','sort']) if ($(name).value && !(name === 'sort' && $(name).value === 'signal')) params.set(name, $(name).value);
  if (selectedStatus) params.set('status', selectedStatus);
  if (selectedObservation) params.set('report', selectedObservation);
  history.replaceState(null, '', `?${params}`);
}
function setView(next, update = true) {
  view = ['reports','benchmarks','methodology'].includes(next) ? next : 'reports';
  for (const name of ['reports','benchmarks','methodology']) $(`${name}-view`).hidden = name !== view;
  document.querySelectorAll('nav [data-view]').forEach(a => {a.classList.toggle('active', a.dataset.view === view); a.setAttribute('aria-current', a.dataset.view === view ? 'page' : 'false');});
  $('view-title').textContent = {reports:'Find the signal.',benchmarks:'Measure what matters.',methodology:'Every result has a contract.'}[view];
  if (update) setURL();
}
function filteredRows() {
  if (!report) return [];
  const needle = $('search').value.trim().toLowerCase();
  const rows = report.results.filter(r => (!needle || (needle.startsWith("op:") ? r.operation.toLowerCase() === needle.slice(3).trim() : `${r.name} ${r.case_id} ${r.observation_id}`.toLowerCase().includes(needle))) &&
    (!$('category').value || r.category === $('category').value) && (!$('dtype').value || r.inputs.x.dtype === $('dtype').value) &&
    (!$('optimization').value || r.optimization === $('optimization').value) &&
    (!selectedStatus || (selectedStatus === 'execution' ? ['execution_error','unsupported'].includes(r.status) : r.status === selectedStatus)));
  return rows.sort((a,b) => $('sort').value === 'name' ? a.name.localeCompare(b.name) : $('sort').value === 'error' ?
    (b.comparison?.max_abs ?? -1) - (a.comparison?.max_abs ?? -1) : severity[a.status] - severity[b.status] || a.name.localeCompare(b.name));
}
function renderRows() {
  const rows = filteredRows(); page = Math.min(page, Math.max(0, Math.ceil(rows.length/PAGE_SIZE)-1));
  $('results').innerHTML = rows.length ? rows.slice(page*PAGE_SIZE, (page+1)*PAGE_SIZE).map(r => `<tr><td><div class="op-name"><span class="op-icon">${icons[r.category] || '·'}</span>${esc(r.operation)}</div><span class="case-name">${esc(r.variant)} · ${esc(r.case_id.slice(0,8))}</span></td><td><span class="dtype-tag">${esc(r.inputs.x.dtype)}</span><span class="shape">[${r.inputs.x.shape.join(' × ')}]</span></td><td><span class="opt-label">${r.optimization === 'all' ? 'All enabled' : 'Disabled'}</span></td><td><span class="error-number">${r.comparison?.nonfinite ? 'Non-finite' : fmt(r.comparison?.max_abs)}</span></td><td>${badge(r.status)}</td><td><button class="detail-link" data-observation="${r.observation_id}" aria-label="Inspect ${esc(r.name)} ${esc(r.optimization)}">↗</button></td></tr>`).join('') : '<tr><td colspan="6" class="empty">No evaluations match these filters. Try another operation or reset filters.</td></tr>';
  $('result-count').textContent = rows.length ? `${page*PAGE_SIZE+1}–${Math.min((page+1)*PAGE_SIZE,rows.length)} of ${rows.length} evaluations` : '0 matching evaluations';
  $('page-number').textContent = `${page+1} / ${Math.max(1,Math.ceil(rows.length/PAGE_SIZE))}`;
  $('previous').disabled = page === 0; $('next').disabled = (page+1)*PAGE_SIZE >= rows.length;
  document.querySelectorAll('[data-status]').forEach(b => {b.classList.toggle('selected', b.dataset.status === selectedStatus); b.setAttribute('aria-pressed', String(b.dataset.status === selectedStatus));});
}
function renderBenchmarks() {
  const operations = [...new Set(report.results.map(r=>r.operation))].sort();
  const median = values => {const sorted = values.sort((a,b)=>a-b);return sorted.length ? (sorted[Math.floor((sorted.length-1)/2)] + sorted[Math.ceil((sorted.length-1)/2)])/2 : null;};
  $('coverage-chart').innerHTML = operations.map(op => {
    const rows = report.results.filter(r=>r.operation === op);
    return `<div class="chart-row"><span>${esc(op)}</span><div class="chart-track">${['pass','divergence','invariant_violation','execution'].map(status=>{
      const count = rows.filter(r=>status === 'execution' ? ['execution_error','unsupported'].includes(r.status) : r.status === status).length;
      return count ? `<button class="${status}" style="width:${100*count/rows.length}%" data-chart-op="${esc(op)}" data-chart-status="${status}" title="${esc(op)}: ${count} ${statuses[status] || 'execution issues'}" aria-label="Filter ${esc(op)}: ${count} ${statuses[status] || 'execution issues'}"></button>` : '';
    }).join('')}</div><span>${rows.length}</span></div>`;
  }).join('');
  $('timings').innerHTML = operations.map(op => {
    const rows = report.results.filter(r=>r.operation === op && r.timings.ort && r.timings.reference);
    return `<tr><td>${esc(op)}</td><td>${rows.length}</td><td class="mono">${fmt(median(rows.map(r=>r.timings.ort.median_ms)))} ms</td><td class="mono">${fmt(median(rows.map(r=>r.timings.reference.median_ms)))} ms</td></tr>`;
  }).join('');
}
function openDetail(id) {
  const r = report.results.find(r=>r.observation_id === id); if (!r) return;
  selectedObservation = id; setURL();
  const c = r.comparison || {}, failed = r.checks.filter(x=>!x.passed), command = `python -m gero_stability.cli replay reports/local/artifacts/${r.case_id} --optimization ${r.optimization} --rtol ${r.tolerance.rtol} --atol ${r.tolerance.atol}`;
  const downloads = [['bundle','↓ Reproducer ZIP'],['model','↓ ONNX graph'],['inputs','↓ Inputs NPZ'],['outputs','↓ Outputs NPZ']].map(([key,label])=>`<a class="button outline" href="${safePath(r.artifacts[key])}" download>${label}</a>`).join('');
  $('detail-content').innerHTML = `<h2 id="detail-title">${esc(r.operation)}</h2><div class="detail-subtitle">${esc(r.variant)} / ${esc(r.inputs.x.dtype)} / [${r.inputs.x.shape.join(' × ')}] / ${esc(r.optimization)} optimization</div>${badge(r.status)}<div class="detail-metrics"><div><span>Max abs. (finite pairs)</span><strong>${fmt(c.max_abs)}</strong></div><div><span>Max rel. (finite pairs)</span><strong>${fmt(c.max_rel)}</strong></div><div><span>Max ULP distance</span><strong>${c.max_ulp == null ? '—' : esc(c.max_ulp)}</strong></div></div><p class="detail-note">${c.mismatched ?? '—'} / ${c.elements ?? '—'} elements exceed tolerance. Non-finite pairs: ${c.nonfinite ?? '—'}.<br><code>atol = ${esc(r.tolerance.atol)} · rtol = ${esc(r.tolerance.rtol)}</code><br>Baseline result: <strong>${esc(r.regression.replaceAll('_',' '))}</strong>.</p>${r.status !== 'pass' ? '<p class="note-alert">Locally observed candidate. Backend disagreement alone does not establish an ONNX Runtime bug. Review the contract, input conditioning and reference behavior before publication.</p>' : ''}<div class="detail-downloads">${downloads}</div><p class="detail-hash">Bundle SHA-256<br>${esc(r.artifacts.bundle_sha256)}</p><h3>Mathematical checks <span class="small">${r.checks.length-failed.length}/${r.checks.length} passed</span></h3>${[...failed,...r.checks.filter(x=>x.passed)].map(ch=>`<div class="check-row"><span class="check-result ${ch.passed ? '' : 'fail'}">${ch.passed ? '✓' : '×'}</span><code>${esc(ch.name)}</code><span class="backend">${esc(ch.backend)}</span></div>`).join('')}${r.errors.length ? `<details open><summary>Execution diagnostics</summary><pre class="raw-json">${esc(JSON.stringify(r.errors,null,2))}</pre></details>` : ''}<div class="copy-row"><h3>Replay this graph</h3><button id="copy-command" class="button outline">Copy command</button></div><pre class="detail-code">${esc(command)}</pre><p class="detail-note">Run from the installed project, or unzip the reproducer and replace the artifact directory above. The replay verifies SHA-256 digests before evaluation.</p><details><summary>Environment & raw observation</summary><pre class="raw-json">${esc(JSON.stringify({environment:report.environment,observation:r},null,2))}</pre></details><p class="detail-hash">Observation ${r.observation_id}</p>`;
  if (!$('detail').open) $('detail').showModal();
  $('copy-command').onclick = async () => {try {await navigator.clipboard.writeText(command);$('copy-command').textContent='Copied';} catch {$('copy-command').textContent='Select command below';}};
}
document.addEventListener('click', e => {
  const nav = e.target.closest('[data-view]');
  if (nav) {e.preventDefault(); setView(nav.dataset.view); if (!nav.closest('nav')) $('explore').scrollIntoView();}
  const status = e.target.closest('[data-status]');
  if (status) {selectedStatus = status.dataset.status;page=0;renderRows();setURL();}
  const detail = e.target.closest('[data-observation]'); if (detail) openDetail(detail.dataset.observation);
  const bar = e.target.closest('[data-chart-op]');
  if (bar) {for (const name of ['category','dtype','optimization']) $(name).value='';$('search').value='op:'+bar.dataset.chartOp;selectedStatus=bar.dataset.chartStatus;page=0;setView('reports');renderRows();}
});
for (const name of ['search','category','dtype','optimization','sort']) $(name).addEventListener(name==='search'?'input':'change',()=>{page=0;renderRows();setURL();});
$('previous').onclick=()=>{page--;renderRows();};$('next').onclick=()=>{page++;renderRows();};
$('reset').onclick=()=>{for(const id of ['search','category','dtype','optimization']) $(id).value='';$('sort').value='signal';selectedStatus='';page=0;renderRows();setURL();};
$('close-detail').onclick=()=>$('detail').close();
$('detail').addEventListener('close',()=>{selectedObservation=null;setURL();});
document.addEventListener('keydown',e=>{if(e.key==='/' && !['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName) && !$('detail').open){e.preventDefault();setView('reports');$('search').focus();}});
async function init() {
  const params = new URLSearchParams(location.search);
  for (const name of ['search','category','dtype','optimization','sort']) if(params.has(name)) $(name).value=params.get(name);
  if (!$('sort').value) $('sort').value='signal';
  selectedStatus=params.get('status')||'';setView(params.get('view'),false);
  try {
    const response=await fetch('data/latest.json');if(!response.ok)throw new Error(`HTTP ${response.status}`);
    report=await response.json();if(report.schema_version!==1||!Array.isArray(report.results))throw new Error('Unsupported report schema');
    const env=report.environment, summary=report.summary;
    $('ort-version').textContent=`v${env.onnxruntime} · CPU · ${env.machine}`;
    $('ref-version').textContent=`ONNX ${env.onnx} · optimized=False`;
    $('stat-total').textContent=report.results.length;
    $('stat-graphs').textContent=`${new Set(report.results.map(r=>r.case_id)).size} graphs · ${report.config.optimizations.length} optimization levels`;
    $('stat-pass').textContent=summary.pass||0;$('stat-divergence').textContent=summary.divergence||0;
    $('stat-other').textContent=(summary.invariant_violation||0)+(summary.execution_error||0)+(summary.unsupported||0);
    if(report.baseline)$('stat-regression').textContent=`${report.results.filter(r=>r.regression==='regression').length} baseline regressions`;
    $('run-label').textContent=`${new Date(report.created_at).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'})} · seed ${report.config.seed}`;
    $('environment-label').textContent=`CPU / ${env.machine} · ONNX ${env.onnx} · ORT ${env.onnxruntime}`;
    renderRows();renderBenchmarks();
    if(params.get('report'))openDetail(params.get('report'));
  } catch(error) {
    $('load-error').hidden=false;$('load-error').textContent=`Run data could not be loaded (${error.message}). Generate a report and run: python -m gero_stability.cli export reports/local/latest.json --site site. Serve this directory over HTTP.`;
    $('results').innerHTML='<tr><td colspan="6" class="empty">No measured data available. Export a completed run to populate this workspace.</td></tr>';
    $('result-count').textContent='No run loaded';$('previous').disabled=true;$('next').disabled=true;
  }
}
init();
