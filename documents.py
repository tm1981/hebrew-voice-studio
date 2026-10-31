"""Long documents: extract text from TXT/DOCX/PDF and split it into chapters, read in a
background job (one library item per chapter) so the UI can show progress.

Chapters: lines that look like headings ("פרק 3", "Chapter 3", "# Title") start a new
chapter; text without headings is cut into parts of about CHAPTER_CHARS at paragraph ends.
"""
import io
import re
import threading
import time
import uuid
import zipfile

MAX_DOCUMENT_CHARS = 300_000
CHAPTER_CHARS = 3000
HEADING = re.compile(r"^\s*(?:פרק\s+\S+|chapter\s+\S+|#{1,3}\s+.+)\s*$", re.IGNORECASE)


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".docx"):
        import docx

        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if name.endswith((".txt", ".md")):
        for encoding in ("utf-8-sig", "cp1255", "utf-16"):  # Hebrew Windows files are often cp1255
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Couldn't read the text file's encoding (tried UTF-8, Windows-1255, UTF-16).")
    raise ValueError("Upload a .txt, .md, .docx or .pdf file.")


def split_chapters(text: str) -> list[tuple[str, str]]:
    """-> [(title, text)]"""
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r", "")).strip()
    lines = text.split("\n")
    if any(HEADING.match(l) for l in lines):
        chapters, title, body = [], None, []
        for line in lines:
            if HEADING.match(line):
                if "".join(body).strip():
                    chapters.append((title or "פתיחה", "\n".join(body).strip()))
                title, body = line.strip().lstrip("#").strip(), []
            else:
                body.append(line)
        if "".join(body).strip():
            chapters.append((title or "פתיחה", "\n".join(body).strip()))
        return chapters
    parts, current = [], ""
    for paragraph in (p.strip() for p in text.split("\n\n") if p.strip()):
        if current and len(current) + len(paragraph) > CHAPTER_CHARS:
            parts.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return [(f"חלק {i}", p) for i, p in enumerate(parts, 1)]


class Jobs:
    """Background document readings, one at a time per job thread."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, title: str, chapters: list[tuple[str, str]], read_chapter) -> dict:
        """read_chapter(title, text, group_id) -> library meta; runs in a background thread."""
        job_id = uuid.uuid4().hex[:10]
        job = {"id": job_id, "title": title, "status": "running", "done": 0, "total": len(chapters),
               "chapters": [t for t, _ in chapters], "items": [], "error": None, "started": time.time()}
        self.jobs[job_id] = job

        def run():
            try:
                for chapter_title, text in chapters:
                    if job["status"] == "cancelled":
                        return
                    meta = read_chapter(chapter_title, text, job_id)
                    job["items"].append(meta["id"])
                    job["done"] += 1
                job["status"] = "done"
            except Exception as e:  # report it in the job; the server keeps running
                job["status"], job["error"] = "failed", str(e)

        threading.Thread(target=run, daemon=True, name=f"document-{job_id}").start()
        return job

    def get(self, job_id: str) -> dict:
        return self.jobs[job_id]

    def cancel(self, job_id: str) -> dict:
        job = self.jobs[job_id]
        if job["status"] == "running":
            job["status"] = "cancelled"  # stops after the chapter in progress
        return job


def zip_chapters(paths: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # MP3 doesn't compress further
        for name, data in paths:
            z.writestr(name, data)
    return buf.getvalue()
