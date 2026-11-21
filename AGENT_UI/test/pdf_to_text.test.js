const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const { rust, bridge, rustFn } = require('./shell_sources')

const ROOT = path.join(__dirname, '..', 'ui')
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8')
const renderer = fs.readFileSync(path.join(ROOT, 'renderer.js'), 'utf8')

test('Electron exposes a dedicated PDF to MD tab with opt-in no-think page-by-page Markdown', () => {
  assert.match(html, /id="btn-pdf"/)
  assert.match(html, /id="left-pdf"/)
  assert.match(html, /id="pdf-rewrite-thai" type="checkbox"/)
  assert.match(html, /local LLM[\s\S]*no-think/)
  assert.match(html, /deterministic cleanup[\s\S]*ภาษาไทย/)
  assert.doesNotMatch(html, /id="pdf-rewrite-thai"[^>]*checked/)
})

test('renderer wires the PDF tab without routing through chat', () => {
  assert.match(renderer, /on\('#btn-pdf', \(\) => togglePanel\('pdf'\)\)/)
  assert.match(renderer, /on\('#pdf-start-btn', startPdfToText\)/)
  assert.match(renderer, /async function startPdfToText\(\)/)
  assert.match(renderer, /window\.electronAPI\.pdfToTextStart\(pdfTextState\.rewriteThai\)/)
  assert.match(renderer, /window\.electronAPI\.pdfToTextStatus\(pdfTextState\.jobId\)/)
  assert.match(renderer, /function openPdfTextOutput\(\)/)
  const pdfStart = renderer.slice(renderer.indexOf('async function startPdfToText()'), renderer.indexOf('function openPdfTextOutput()'))
  assert.doesNotMatch(pdfStart, /wsSend\(\{\s*type:\s*'query'/)
})

test('the bridge keeps PDF IPC narrow and the Rust shell streams authenticated bytes', () => {
  assert.match(bridge, /pdfToTextStart: function \(rewriteThai\) \{ return invoke\('pdf_to_text_start'/)
  assert.match(bridge, /pdfToTextStatus: function \(jobId\) \{ return invoke\('agent_job', \{ route: '\/pdf-to-text\/status'/)
  const start = rustFn('pdf_to_text_start')
  assert.match(start, /pick_one\(&app, [^)]*\("PDF", &\["pdf"\]\)/)
  assert.match(rust, /reqwest::Body::from\(file\)/)
  assert.match(rust, /\.header\("X-Auth-Token", token\)/)
  assert.match(start, /"X-Rewrite-Thai", if rewrite_thai \{ "1" \} else \{ "0" \}/)
})
