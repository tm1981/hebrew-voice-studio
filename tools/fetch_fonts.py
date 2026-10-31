"""Downloads the web UI's default fonts into web/fonts, so the page needs no internet.
Run once; the result is part of the project. Fonts picked in the UI's font menu (other
than the defaults) still load from Google Fonts when chosen. All are SIL Open Font License.

Usage:  python tools/fetch_fonts.py
"""
import re
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "web" / "fonts"
FAMILIES = {"Noto Serif Hebrew": "400;500", "Heebo": "400;500;600"}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    query = "&".join(f"family={name.replace(' ', '+')}:wght@{w}" for name, w in FAMILIES.items())
    css = requests.get(f"https://fonts.googleapis.com/css2?{query}&display=swap", headers={"User-Agent": UA}, timeout=30).text
    for url in sorted(set(re.findall(r"url\((https://fonts\.gstatic\.com/[^)]+\.woff2)\)", css))):
        name = url.rsplit("/", 1)[1]
        (OUT / name).write_bytes(requests.get(url, timeout=60).content)
        css = css.replace(url, name)
    (OUT / "fonts.css").write_text(css, encoding="utf-8")
    print(f"{len(list(OUT.glob('*.woff2')))} font files in {OUT}")


if __name__ == "__main__":
    main()
