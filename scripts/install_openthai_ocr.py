# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Install the OpenThai2.1-OCR-Text weights (the optional OCR engine of Settings -> OCR).

    python scripts/install_openthai_ocr.py [target_dir]

Downloads the five model files (about 4.2 MB) from Hugging Face into target_dir (default
config.OPENTHAI_OCR_DIR = ~/Models/OpenThai2.1-OCR-Text).  Every file is checked against the SHA-256 audited
on 2026-11-15 before it is moved into place, so a changed upstream file fails closed instead of being used.
Stdlib only.  The Python packages the engine needs (numpy, opencv, onnxruntime, pyclipper) are not installed
here: pip install pyclipper into the agent's env if it is the one missing (the others are already there).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request

BASE = "https://huggingface.co/iapp/OpenThai2.1-OCR-Text-20261118/resolve/main/"
FILES = {
    "det.onnx": "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
    "rec.onnx": "82ec89533346faadcb30df96703dad2dd7521de65709e60d83fd9745b8968124",
    "rec_dict.txt": "259ef2636cec0eeae225ae9fd9a63fa9305285650c22d7ec34fe8a82b24c2f1a",
    "config.json": "f22dcf63fe64d88a8f8f066b1168e0e373ec008ea9d67568c29f0b8221fa1d31",
    "rec2.onnx": "6ec0f96f5ade665fc7eb33a9800ed055bc220870cbf963cf97c4750e64cd27ce",
    "det.yml": "3ac018be6f97499a08faa3bbdeb33640968d9307f6736d152902747a9f259593",
}
REQUIRED = tuple(FILES)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(target: str) -> list[str]:
    """Fetch every missing or wrong file; return the names that were downloaded.  Raises RuntimeError on a hash mismatch."""
    os.makedirs(target, exist_ok=True)
    fetched = []
    for name, want in FILES.items():
        dest = os.path.join(target, name)
        if os.path.isfile(dest) and _sha256(dest) == want:
            continue
        fd, tmp = tempfile.mkstemp(prefix=".dl-", dir=target)
        os.close(fd)
        try:
            with urllib.request.urlopen(BASE + name, timeout=120) as response, open(tmp, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            got = _sha256(tmp)
            if got != want:
                raise RuntimeError(f"{name}: SHA-256 mismatch (got {got}, expected {want}); upstream changed or the download is corrupt")
            os.replace(tmp, dest)
            fetched.append(name)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return fetched


def main(argv: list[str]) -> int:
    if len(argv) > 2 or (len(argv) == 2 and argv[1] in ("-h", "--help")):
        print(__doc__)
        return 0
    if len(argv) == 2:
        target = os.path.expanduser(argv[1])
    else:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import config  # noqa: E402  (target default lives in config so the agent and the installer agree)
        target = config.OPENTHAI_OCR_DIR
    try:
        fetched = install(target)
    except (OSError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(f"OpenThai OCR weights ready in {target}" + (f" (downloaded: {', '.join(fetched)})" if fetched else " (already present, hashes verified)"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
