const test = require('node:test')
const assert = require('node:assert')
const {
  summarizeWaitingQuery,
  toolDisplay,
  friendlyProgress,
  normalizePhase,
} = require('../lib/waiting_status')

test('summarizeWaitingQuery detects web freshness intent', () => {
  assert.match(summarizeWaitingQuery('ราคาทองวันนี้ล่าสุด'), /ข้อมูลล่าสุดจากเว็บ/)
})

test('summarizeWaitingQuery detects workspace file intent', () => {
  assert.match(summarizeWaitingQuery('/tmp/report.pdf ช่วยอ่าน'), /workspace/)
})

test('toolDisplay gives friendly web_search label/sub', () => {
  const out = toolDisplay('web_search', '"Fed rate cut today"')
  assert.strictEqual(out.label, 'กำลังค้นเว็บ…')
  assert.match(out.sub, /Fed rate cut today/)
})

test('toolDisplay gives friendly rag_search fallback text', () => {
  const out = toolDisplay('rag_search', '')
  assert.strictEqual(out.label, 'กำลังค้น knowledge base…')
  assert.match(out.sub, /KB/)
})

test('friendlyProgress rewrites cache hit and batch counters', () => {
  assert.strictEqual(
    friendlyProgress('web_search', '[2/5] summary HIT (query-aware): https://x.com'),
    'ใช้ cache สรุปเว็บ 2/5'
  )
  assert.strictEqual(
    friendlyProgress('web_search', 'summarized 3 URLs in 1.2s'),
    'ได้ข้อมูลครบจาก 3 แหล่ง กำลังสรุปต่อ…'
  )
  assert.strictEqual(
    friendlyProgress('batch_browse', 'HTTP 2/4: https://example.com/a'),
    'กำลังเปิดเว็บ 2/4'
  )
})

test('friendlyProgress rewrites rag coverage and scrape fallback', () => {
  assert.strictEqual(
    friendlyProgress('rag_search', 'found 4 chunk(s)'),
    'เจอ chunk ที่เกี่ยวข้อง 4 ส่วน'
  )
  assert.strictEqual(
    friendlyProgress('scrape_table', 'ไม่พบ <table> — ลอง ARIA role fallback'),
    'ไม่พบตาราง HTML ตรง ๆ กำลังลองอ่านจาก ARIA role'
  )
  assert.strictEqual(
    friendlyProgress('browse_url', 'cached (123 raw + 45 summary)'),
    'อ่านเว็บแล้วและเก็บ cache แล้ว กำลังสรุปต่อ…'
  )
})

test('normalizePhase localizes generic phases and neutral continue state', () => {
  assert.strictEqual(normalizePhase('thinking…', ''), 'กำลังวิเคราะห์คำถาม…')
  assert.strictEqual(normalizePhase('synthesizing…', ''), 'กำลังจัดรูปคำตอบสุดท้าย…')
  assert.strictEqual(normalizePhase('executing…', 'web_search'), 'ได้ผลจาก web_search แล้ว กำลังทำต่อ…')
  assert.strictEqual(normalizePhase('custom label', 'web_search'), 'custom label')
})
