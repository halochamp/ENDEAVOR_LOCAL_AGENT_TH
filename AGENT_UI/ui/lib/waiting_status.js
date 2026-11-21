// waiting_status.js — pure UI-side translators for waiting/status text.
//
// Loaded as a plain <script> in index.html so renderer.js can call these globals,
// and via require() in node --test. No DOM access here.

function summarizeWaitingQuery(query) {
  const q = String(query || '').trim().replace(/\s+/g, ' ')
  const low = q.toLowerCase()
  if (!q) return 'กำลังวิเคราะห์คำถามและเลือกเครื่องมือที่เหมาะสม'
  if (/https?:\/\//.test(low)) return 'กำลังวิเคราะห์ว่าจะอ่านจากเว็บตรงหรือค้นเพิ่มก่อน'
  if (q.includes('/') || q.includes('\\') || /\.(md|txt|pdf|docx|xlsx|xls|csv|json|py)\b/i.test(low)) {
    return 'กำลังวิเคราะห์ว่าโจทย์นี้ต้องอ่านไฟล์ไหนใน workspace'
  }
  if (/(วันนี้|ล่าสุด|อัปเดต|update|current|latest|news|ราคา|หุ้น|ดอกเบี้ย)/i.test(low)) {
    return 'กำลังประเมินว่าต้องใช้ข้อมูลล่าสุดจากเว็บหรือไม่'
  }
  if (/(จากเอกสาร|ต้นฉบับ|kb|knowledge|บทวิเคราะห์|สรุป|กองทุน|งบ|retirement|portfolio|valuation)/i.test(low)) {
    return 'กำลังประเมินว่าแหล่ง MCP หรือเอกสารต้นฉบับเหมาะกับคำถามนี้'
  }
  return 'กำลังวิเคราะห์คำถามและเลือกเครื่องมือที่เหมาะสม'
}

function toolDisplay(name, detail) {
  const d = String(detail || '').trim()
  const withDetail = (label, fallback) => ({
    label,
    sub: d || fallback || '',
    activityDetail: d || fallback || '',
  })
  switch (name) {
    case 'create_plan': return withDetail('กำลังวางแผน…', 'กำลังแตกงานเป็นขั้นตอนก่อนลงมือทำ')
    case 'web_search': return withDetail('กำลังค้นเว็บ…', 'กำลังค้นเว็บเพื่อหาข้อมูลภายนอกหรือล่าสุด')
    case 'browse_url': return withDetail('กำลังอ่านเว็บ…', 'กำลังอ่านหน้าเว็บหรือทำงานกับ URL ตามโหมดที่เลือก')
    case 'recall_web': return withDetail('กำลังดึงเว็บจาก cache…', 'กำลังดึงหน้าเว็บที่เคยแคชไว้เพื่อลดการโหลดซ้ำ')
    case 'read_file': return withDetail('กำลังอ่านไฟล์…', 'กำลังอ่านไฟล์ต้นฉบับใน workspace')
    case 'edit': return withDetail('กำลังแก้ไฟล์…', 'กำลังสร้างหรือแก้ไฟล์ผ่าน edit access guard')
    case 'plot': return withDetail('กำลังสร้างกราฟ…', 'กำลังสร้างกราฟหรือภาพผลลัพธ์จากข้อมูลที่มี')
    case 'python_exec': return withDetail('กำลังคำนวณด้วย Python…', 'กำลังคำนวณหรือแปลงข้อมูลด้วย Python')
    case 'bash': return withDetail('กำลังรันคำสั่งระบบ…', 'กำลังเช็ค environment หรือรันคำสั่งระบบที่จำเป็น')
    case 'read_image': return withDetail('กำลังอ่านภาพ…', 'กำลังอ่านภาพเพื่อตอบคำถามเฉพาะจุดจากรูปนี้')
    case 'mcp_list_tools': return withDetail('กำลังดู MCP tools…', 'กำลังอ่าน catalog และ schema ของ MCP server')
    case 'mcp_call_tool': return withDetail('กำลังเรียก MCP tool…', 'กำลังเรียก tool จาก server ที่เลือกไว้')
    default: return withDetail('กำลังทำงาน…', '')
  }
}

function friendlyProgress(toolName, msg) {
  const text = String(msg || '').trim().replace(/\s+/g, ' ')
  const low = text.toLowerCase()
  if (!text) return ''

  if (toolName === 'web_search') {
    let m = low.match(/fetching (\d+) urls in parallel/)
    if (m) return `กำลังเปิด ${m[1]} แหล่งพร้อมกัน`
    m = low.match(/\[(\d+)\/(\d+)\] summary hit/)
    if (m) return `ใช้ cache สรุปเว็บ ${m[1]}/${m[2]}`
    m = low.match(/\[(\d+)\/(\d+)\] raw hit/)
    if (m) return `มี raw cache แล้ว กำลังสรุปใหม่ ${m[1]}/${m[2]}`
    m = low.match(/\[(\d+)\/(\d+)\] queued for summarizing/)
    if (m) return `กำลังเตรียมสรุปเว็บ ${m[1]}/${m[2]}`
    m = low.match(/summarized (\d+) urls/)
    if (m) return `ได้ข้อมูลครบจาก ${m[1]} แหล่ง กำลังสรุปต่อ…`
    if (low.includes('time-sensitive query detected')) return 'คำถามนี้อิงเวลา จึงจำกัดผลลัพธ์เป็นช่วงล่าสุด'
    if (low.startsWith('search:')) return 'กำลังค้นหาแหล่งข้อมูลบนเว็บ'
    if (low.startsWith('backfill:')) return 'บางเว็บโหลดไม่สำเร็จ กำลังสลับไปใช้แหล่งสำรอง'
  }

  if (toolName === 'browse_url') {
    if (low.includes('summary hit')) return 'หน้าเว็บนี้มี summary cache แล้ว จึงตอบได้เร็วขึ้น'
    if (low.includes('raw hit')) return 'มี raw cache แล้ว กำลังสรุปใหม่ให้ตรงคำถาม'
    if (low.includes('cache miss')) return 'ยังไม่มี cache สำหรับหน้านี้ จึงต้อง fetch ใหม่'
    if (low.startsWith('cached (')) return 'อ่านเว็บแล้วและเก็บ cache แล้ว กำลังสรุปต่อ…'
    if (low.startsWith('fetching via jina reader')) return 'กำลังอ่านเว็บผ่านตัวช่วยดึงเนื้อหา'
    if (low.includes('jina returned empty') || low.includes('jina failed')) return 'ตัวอ่านเว็บทางลัดใช้ไม่ได้ กำลัง fallback ไปวิธีอื่น'
    if (low.includes('direct fetch ok')) return 'อ่านเนื้อหาเว็บได้ตรง ๆ โดยไม่ต้องผ่านตัวช่วยเพิ่ม'
    let m = low.match(/http (\d+)\/(\d+):/)
    if (m) return `กำลังเปิดเว็บ ${m[1]}/${m[2]}`
    if (low.includes('web budget exhausted')) return 'งบเว็บของ turn นี้ใกล้หมด จึงใช้เฉพาะข้อมูลที่ cache ไว้'
    m = low.match(/web budget: capping fetch to (\d+)\/(\d+) urls/)
    if (m) return `จำกัดการ fetch เหลือ ${m[1]}/${m[2]} URLs ตามงบเว็บที่เหลือ`
    if (low.startsWith('navigating')) return 'กำลังเปิดหน้าเว็บและรอให้ตารางโหลด'
    if (low.startsWith('page loaded')) return 'หน้าเว็บโหลดแล้ว กำลังหาโครงตาราง'
    if (low.includes('aria role fallback')) return 'ไม่พบตาราง HTML ตรง ๆ กำลังลองอ่านจาก ARIA role'
    if (low.includes('div-grid heuristic')) return 'ไม่พบตารางมาตรฐาน กำลังลองอ่านจาก div-grid'
    const tableMatch = text.match(/table (\d+) \(([^)]+)\): (\d+) rows × (\d+) cols/i)
    if (tableMatch) return `พบตาราง ${tableMatch[1]} แล้ว (${tableMatch[3]} แถว x ${tableMatch[4]} คอลัมน์)`
  }

  if (toolName === 'recall_web') {
    if (low.includes('cache hit')) return 'ใช้ raw cache ของเว็บนี้ได้ทันที'
    if (low.includes('cache miss')) return 'ยังไม่มี raw cache จึงต้องดึงเว็บใหม่หนึ่งรอบ'
  }

  if (toolName === 'read_image') {
    if (low.includes('cache')) return 'ภาพนี้มี cache อยู่แล้ว จึงข้ามขั้นบางส่วนได้'
    return text
  }

  return text
}

function normalizePhase(label, currentToolName) {
  const text = String(label || '').trim()
  if (!text) return text
  if (text === 'thinking…') return 'กำลังวิเคราะห์คำถาม…'
  if (text === 'synthesizing…') return 'กำลังจัดรูปคำตอบสุดท้าย…'
  if (text === 'executing…') {
    return currentToolName ? `ได้ผลจาก ${currentToolName} แล้ว กำลังทำต่อ…` : 'กำลังทำขั้นถัดไป…'
  }
  return text
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { summarizeWaitingQuery, toolDisplay, friendlyProgress, normalizePhase }
}
