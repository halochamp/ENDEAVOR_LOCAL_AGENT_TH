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
    return 'กำลังประเมินว่าควรเริ่มจาก knowledge base หรือเอกสารต้นฉบับ'
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
    case 'rag_search': return withDetail('กำลังค้น knowledge base…', 'กำลังค้น KB เพื่อดึงส่วนที่ตอบคำถามได้ตรงที่สุด')
    case 'rag_ls': return withDetail('กำลังสำรวจ knowledge base…', 'กำลังส่องโครง KB เพื่อเลือก query ที่แม่นขึ้น')
    case 'rag_rebuild_index': return withDetail('กำลัง rebuild knowledge base…', 'กำลัง rebuild ดัชนี KB เพื่อให้ค้นหาได้แม่นขึ้น')
    case 'rag_list_knowledge': return withDetail('กำลังดูภาพรวม knowledge base…', 'กำลังดูภาพรวม knowledge base ว่ามีหัวข้ออะไรอยู่บ้าง')
    case 'web_search': return withDetail('กำลังค้นเว็บ…', 'กำลังค้นเว็บเพื่อหาข้อมูลภายนอกหรือล่าสุด')
    case 'browse_url': return withDetail('กำลังอ่านหน้าเว็บ…', 'กำลังอ่านหน้าเว็บนี้ตรง ๆ เพื่อดึงคำตอบจากแหล่งต้นทาง')
    case 'recall_web': return withDetail('กำลังดึงเว็บจาก cache…', 'กำลังดึงหน้าเว็บที่เคยแคชไว้เพื่อลดการโหลดซ้ำ')
    case 'batch_browse': return withDetail('กำลังอ่านหลายเว็บ…', 'กำลังเทียบหลายหน้าเว็บพร้อมกันแล้วสรุปกลับมาในรอบเดียว')
    case 'fetch_sitemap': return withDetail('กำลังกวาด sitemap…', 'กำลังกวาดโครง URL ของทั้งเว็บเพื่อหาแหล่งข้อมูลให้ครบ')
    case 'scrape_table': return withDetail('กำลังดึงตารางจากเว็บ…', 'กำลังดึงตารางจากหน้าเว็บในรูปแบบที่เอาไปวิเคราะห์ต่อได้')
    case 'read_file': return withDetail('กำลังอ่านไฟล์…', 'กำลังอ่านไฟล์ต้นฉบับใน workspace')
    case 'write_file': return withDetail('กำลังเขียนไฟล์…', 'กำลังเขียนไฟล์ผลลัพธ์ลง workspace')
    case 'edit': return withDetail('กำลังแก้ไฟล์…', 'กำลังแก้ไฟล์เดิมตามผลวิเคราะห์')
    case 'plot': return withDetail('กำลังสร้างกราฟ…', 'กำลังสร้างกราฟหรือภาพผลลัพธ์จากข้อมูลที่มี')
    case 'tool_loop': return withDetail('กำลังทำงานหลายรายการ…', 'กำลังวนทำงานหลายรายการใน tool เดียวเพื่อลดรอบ')
    case 'python_exec': return withDetail('กำลังคำนวณด้วย Python…', 'กำลังคำนวณหรือแปลงข้อมูลด้วย Python')
    case 'bash': return withDetail('กำลังรันคำสั่งระบบ…', 'กำลังเช็ค environment หรือรันคำสั่งระบบที่จำเป็น')
    case 'read_image': return withDetail('กำลังอ่านภาพ…', 'กำลังอ่านภาพเพื่อตอบคำถามเฉพาะจุดจากรูปนี้')
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
  }

  if (toolName === 'recall_web') {
    if (low.includes('cache hit')) return 'ใช้ raw cache ของเว็บนี้ได้ทันที'
    if (low.includes('cache miss')) return 'ยังไม่มี raw cache จึงต้องดึงเว็บใหม่หนึ่งรอบ'
  }

  if (toolName === 'batch_browse') {
    let m = low.match(/http (\d+)\/(\d+):/)
    if (m) return `กำลังเปิดเว็บ ${m[1]}/${m[2]}`
    if (low.includes('web budget exhausted')) return 'งบเว็บของ turn นี้ใกล้หมด จึงใช้เฉพาะข้อมูลที่ cache ไว้'
    m = low.match(/web budget: capping fetch to (\d+)\/(\d+) urls/)
    if (m) return `จำกัดการ fetch เหลือ ${m[1]}/${m[2]} URLs ตามงบเว็บที่เหลือ`
  }

  if (toolName === 'scrape_table') {
    if (low.startsWith('navigating')) return 'กำลังเปิดหน้าเว็บและรอให้ตารางโหลด'
    if (low.startsWith('page loaded')) return 'หน้าเว็บโหลดแล้ว กำลังหาโครงตาราง'
    if (low.includes('aria role fallback')) return 'ไม่พบตาราง HTML ตรง ๆ กำลังลองอ่านจาก ARIA role'
    if (low.includes('div-grid heuristic')) return 'ไม่พบตารางมาตรฐาน กำลังลองอ่านจาก div-grid'
    const m = text.match(/table (\d+) \(([^)]+)\): (\d+) rows × (\d+) cols/i)
    if (m) return `พบตาราง ${m[1]} แล้ว (${m[3]} แถว x ${m[4]} คอลัมน์)`
  }

  if (toolName === 'rag_search') {
    let m = low.match(/searching knowledge base.*\((\d+) query variants\)/)
    if (m) return `กำลังค้น KB ด้วย ${m[1]} query variants`
    m = low.match(/found (\d+) source file/)
    if (m) return `เจอไฟล์ต้นทาง ${m[1]} ไฟล์ใน KB`
    m = low.match(/found (\d+) chunk/)
    if (m) return `เจอ chunk ที่เกี่ยวข้อง ${m[1]} ส่วน`
    if (low.includes('rag error')) return 'KB ยังไม่ให้ผลตรงพอ กำลังเตรียมหาทางอื่น'
  }

  if (toolName === 'tool_loop') {
    let m = text.match(/\[(\d+)\/(\d+)\]/)
    if (m) return `กำลังทำรายการ ${m[1]}/${m[2]}`
    m = low.match(/urls:\s*(\d+)\/(\d+)/)
    if (m) return `รวบรวม URLs ได้ ${m[1]}/${m[2]}`
    return text
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
