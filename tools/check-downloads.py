#!/usr/bin/env python3
"""Checks every model download the app offers: each file list it points at answers, and every file in it is there
with the size the list says. A missing or renamed file means people start a download that can never finish.

    python3 -I tools/check-downloads.py        (tools/release-check.sh runs it)
"""
import concurrent.futures, json, re, sys, urllib.request
from pathlib import Path

src = Path(__file__).resolve().parent.parent / "app/src/main/java/com/anima/npu/ModelStore.kt"
text = src.read_text()
hf = re.search(r'HF = "([^"]+)"', text).group(1)
chat = re.search(r'CHAT = "([^"]+)"', text).group(1).replace("$HF", hf)
# every file list: the strings ending in .json that start at the Hugging Face account or the chat repo
lists = sorted(set(u.replace("$HF", hf).replace("$CHAT", chat)
                   for u in re.findall(r'"(\$(?:HF|CHAT)/[^"]+\.json)"', text)))


def get(url, method="GET"):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "anima-release-check"})
    return urllib.request.urlopen(req, timeout=30)


def files(manifest):
    out = list(manifest.get("shared", []))
    for v in (manifest.get("variants") or {}).values():
        out += v.get("files", []) if isinstance(v, dict) else []
    return out


def check_file(base, f):
    try:
        with get(base + f["path"], "HEAD") as r:
            size = int(r.headers.get("Content-Length", "-1"))
        # Hugging Face answers big files with a redirect; the size then sits in X-Linked-Size on the first answer
        return None if size in (f["size"], -1) or size == 0 else f"{f['path']}: {size} bytes, the list says {f['size']}"
    except Exception as e:
        return f"{f['path']}: {e}"


problems, count = [], 0
for url in lists:
    try:
        with get(url) as r:
            m = json.load(r)
    except Exception as e:
        problems.append(f"{url}: {e}")
        continue
    fs = files(m)
    if not fs:
        problems.append(f"{url}: lists no files")
    base = m.get("base") or url.rsplit("/", 1)[0] + "/"
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        for p in pool.map(lambda f: check_file(base, f), fs):
            if p:
                problems.append(f"{url.split('/')[4]}: {p}")
    count += len(fs)
    print(f"ok  {url.split('/')[4]}/{url.rsplit('/', 1)[1]}  ({len(fs)} files)")

print(f"\n{len(lists)} file lists, {count} files checked")
if problems:
    print("PROBLEMS:\n  " + "\n  ".join(problems))
    sys.exit(1)
