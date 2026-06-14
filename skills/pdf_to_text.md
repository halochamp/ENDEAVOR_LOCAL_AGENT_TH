## Role
แปลง PDF เป็นไฟล์ข้อความ .txt — รองรับทั้ง PDF แท้ (มีข้อความ copy ได้) และ PDF scan (ภาพถ่าย)
ตรวจจับประเภทอัตโนมัติก่อนเสมอ แล้วเลือก path ที่เหมาะสม

## Detection Logic
- **Text PDF** → `page.get_text()` ได้ข้อความ ≥ 50 chars/หน้าเฉลี่ย (3 หน้าแรก) → native path
- **Scan PDF** → text น้อยกว่านั้น → OCR path

## Workflow
1. รับ path ของไฟล์ PDF จาก user ตาม priority นี้:
   - user ระบุ path มาแล้ว → ใช้เลย
   - user ไม่ระบุ → ใช้ python_exec หา PDF ใน workspace ก่อน:
     ```python
     import glob, os
     import config; workspace = config.WORKSPACE
     pdfs = sorted(set(
         glob.glob(os.path.join(workspace, "**/*.pdf"), recursive=True) +
         glob.glob(os.path.join(workspace, "*.pdf"))
     ))
     print(pdfs)
     ```
   - พบ 1 ไฟล์ → ใช้ไฟล์นั้นเลย บอก user ว่าใช้ไฟล์ไหน
   - พบ หลายไฟล์ → แสดงรายชื่อ ถาม user ว่าต้องการไฟล์ไหน (1 คำถาม)
   - ไม่พบเลย → ถาม user ขอ path (1 คำถาม)
2. ใช้ python_exec รันโค้ดด้านล่าง — auto-detect → เลือก path → เขียนไฟล์ทีละหน้า

```python
import fitz, uuid, os, sys, urllib.request, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools._ocr import read_text as ocr_read

PDF_PATH  = "<ใส่ path PDF>"
MAX_PAGES = 50
LLM_URL   = os.getenv("MLX_BASE_URL", "http://localhost:8080/v1") + "/chat/completions"
LLM_MODEL = os.getenv("V2_MODEL", "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit")

NATIVE_THRESHOLD = 50    # chars/page เฉลี่ย → ถือว่าเป็น text PDF


# ── helpers ──────────────────────────────────────────────────────────────

def _llm_call(prompt: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(LLM_URL, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.loads(r.read())
        return res["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None  # ใช้ raw แทน


def llm_polish_ocr(raw: str, page_num: int) -> str:
    """เกลาข้อความจาก OCR — แก้ตัวอักษรผิด ช่องว่างไม่ถูก"""
    prompt = (
        f"ข้อความต่อไปนี้มาจาก OCR หน้า {page_num} ของเอกสาร "
        "อาจมีตัวอักษรผิดพลาด คำขาดหาย หรือเว้นวรรคไม่ถูก "
        "ช่วยแก้ไขให้ถูกต้องและอ่านง่าย โดยคงความหมายเดิมไว้ทุกคำ "
        "ห้ามเพิ่มหรือตัดเนื้อหา ตอบเฉพาะข้อความที่แก้แล้ว:\n\n" + raw
    )
    return _llm_call(prompt) or raw


def llm_check_native(raw: str, page_num: int) -> str:
    """ตรวจข้อความ native PDF — แก้ typo, ligature, hyphenation ท้ายบรรทัด"""
    prompt = (
        f"ข้อความต่อไปนี้ extract จาก PDF หน้า {page_num} โดยตรง "
        "อาจมีปัญหา: คำขาดตรงกลาง (hyphen ท้ายบรรทัด เช่น 'im-portant'), "
        "ligature ที่แตกเป็นตัวอักษรผิด (เช่น 'ﬁ'→'fi'), "
        "หรือ typo เล็กน้อย "
        "ช่วยแก้ไขให้ถูกต้องและอ่านง่าย โดยคงความหมายเดิมไว้ทุกคำ "
        "ห้ามเพิ่มหรือตัดเนื้อหา ตอบเฉพาะข้อความที่แก้แล้ว:\n\n" + raw
    )
    return _llm_call(prompt) or raw


def detect_pdf_type(doc) -> str:
    """ตรวจ 3 หน้าแรก ถ้าเฉลี่ย ≥ NATIVE_THRESHOLD chars/หน้า → 'text', ไม่งั้น → 'scan'"""
    sample = min(3, len(doc))
    total  = sum(len(doc[i].get_text().strip()) for i in range(sample))
    avg    = total / sample if sample else 0
    return "text" if avg >= NATIVE_THRESHOLD else "scan"


# ── main ─────────────────────────────────────────────────────────────────

doc     = fitz.open(PDF_PATH)
n_total = len(doc)
n       = min(n_total, MAX_PAGES)
mode    = detect_pdf_type(doc)

suffix   = "_text.txt" if mode == "text" else "_ocr.txt"
OUT_PATH = PDF_PATH.rsplit(".", 1)[0] + suffix

print(f"PDF: {n_total} หน้า, จะประมวลผล {n} หน้า")
print(f"ประเภท: {'📄 Text PDF (native extract)' if mode == 'text' else '🖼 Scan PDF (OCR)'}")
print(f"Output: {OUT_PATH}\n")

# สร้างไฟล์ใหม่พร้อม header (เขียนทับถ้ามีอยู่แล้ว)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"# {os.path.basename(PDF_PATH)}\n")
    f.write(f"# {n} หน้า | {'Text PDF' if mode == 'text' else 'Scan PDF'}\n")
    f.write("=" * 50 + "\n\n")

SEP_THICK = "═" * 50
SEP_THIN  = "─" * 50

skipped = 0
for i in range(n):
    print(f"  หน้า {i+1}/{n}", end=" ", flush=True)
    page = doc[i]

    if mode == "text":
        # ── native path ──────────────────────────────────────────
        raw = page.get_text().strip()
        if not raw:
            print("— ว่าง ข้าม")
            skipped += 1
            # append placeholder เพื่อรักษาลำดับหน้า
            with open(OUT_PATH, "a", encoding="utf-8") as f:
                f.write(f"{SEP_THICK}\n หน้า {i+1} / {n}\n{SEP_THICK}\n")
                f.write("[หน้านี้ไม่มีข้อความ]\n\n")
            continue
        print(f"— {len(raw)} chars → LLM check...", end=" ", flush=True)
        polished = llm_check_native(raw, i + 1)

    else:
        # ── OCR path ─────────────────────────────────────────────
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        tmp = f"/tmp/pdf_ocr_{uuid.uuid4().hex[:8]}.png"
        pix.save(tmp)
        lines = ocr_read(tmp)
        os.unlink(tmp)
        if not lines:
            print("— OCR ว่าง ข้าม")
            skipped += 1
            with open(OUT_PATH, "a", encoding="utf-8") as f:
                f.write(f"{SEP_THICK}\n หน้า {i+1} / {n}\n{SEP_THICK}\n")
                f.write("[หน้านี้ OCR ไม่ได้ข้อความ]\n\n")
            continue
        raw = "\n".join(lines)
        print(f"— {len(raw)} chars → LLM polish...", end=" ", flush=True)
        polished = llm_polish_ocr(raw, i + 1)

    # append หน้านี้ลงไฟล์ทันที — 1 หน้า = 1 save
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"{SEP_THICK}\n หน้า {i+1} / {n}\n{SEP_THICK}\n")
        f.write(polished + "\n")
        f.write(SEP_THIN + "\n\n")

    print(f"done ✓ → บันทึกแล้ว ({os.path.getsize(OUT_PATH)//1024} KB)")

doc.close()
print(f"\n✅ เสร็จ: {OUT_PATH}")
print(f"ขนาดไฟล์: {os.path.getsize(OUT_PATH)//1024} KB")
if skipped:
    print(f"หน้าที่ข้าม: {skipped} หน้า (ไม่มีข้อความ)")

with open(OUT_PATH, encoding="utf-8") as f:
    print("\nPreview:\n" + "".join(f.readlines()[:10]))
```

3. บอก user: path ไฟล์ output + ประเภท PDF ที่ตรวจพบ + preview
4. ถ้าบางหน้าว่าง → บอกจำนวนที่ข้ามไป

## Tools allowed
python_exec

## Notes
- **detection threshold:** 50 chars/page เฉลี่ย (3 หน้าแรก) — ข้อความจริงมี 500+ chars/หน้า, ต่ำกว่านี้ถือเป็น scan
- **Text PDF path:** `page.get_text()` → `llm_check_native()` — เร็วกว่า OCR มาก, prompt เน้น hyphen/ligature/typo
- **Scan PDF path:** render PNG 2× → `ocr_read()` → `llm_polish_ocr()` — เหมือนเดิม
- output suffix: `_text.txt` (text PDF) / `_ocr.txt` (scan) — แยกให้ชัด
- LLM ใช้ `enable_thinking: False` — เร็ว 10×, temperature=0.1 คงเนื้อหา
- ถ้า LLM call fail → ใช้ raw text ต่อ (ไม่ stop)
- MAX_PAGES=50 default — user ขอเพิ่มได้
