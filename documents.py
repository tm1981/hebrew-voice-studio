"""Long documents: extract text from TXT/DOCX/PDF and split it into chapters, read in a
background job (one library item per chapter) so the UI can show progress.

Chapters: lines that look like headings ("פרק 3", "Chapter 3", "# Title") start a new
chapter; text without headings is cut into parts of about CHAPTER_CHARS at paragraph ends.
"""
import io
import json
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
    """Background document readings, one thread per job, kept in a file so they survive a restart.

    A job: {"id", "title", "status", "done", "total", "chapters" (titles), "items" (library ids),
    "error", "started"}. status: running, done, cancelled, failed (a chapter failed twice) or
    interrupted (the server stopped while it ran). All but running and done can be resumed from
    the first chapter not read yet; the chapter texts and the reading settings are kept for that.
    """

    KEEP = 20  # jobs remembered
    RESUMABLE = ("failed", "cancelled", "interrupted")

    def __init__(self, path=None, read=None):
        """path: where jobs are saved (None: memory only). read(title, settings, chapter_title,
        text, group) -> library meta; it can be set later, before the first job starts."""
        self.path, self.read = path, read
        self.jobs: dict[str, dict] = {}
        self.work: dict[str, dict] = {}  # id -> {"chapters": [[title, text]], "settings": {...}}
        self.lock = threading.Lock()
        if path and path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.jobs = {j["id"]: j for j in saved["jobs"]}
                self.work = saved["work"]
            except (OSError, ValueError, KeyError, TypeError):
                self.jobs, self.work = {}, {}  # a damaged file loses the list, not the readings
            for job in self.jobs.values():
                if job["status"] == "running":  # the server stopped under it
                    job["status"] = "interrupted"
            self._save()

    def _save(self) -> None:
        with self.lock:
            for old in sorted(self.jobs.values(), key=lambda j: j["started"], reverse=True)[self.KEEP:]:
                self.jobs.pop(old["id"], None)
                self.work.pop(old["id"], None)
            for job_id in [i for i, j in self.jobs.items() if j["status"] == "done"]:
                self.work.pop(job_id, None)  # nothing left to resume
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temp = self.path.with_suffix(".tmp")
                temp.write_text(json.dumps({"jobs": list(self.jobs.values()), "work": self.work}, ensure_ascii=False),
                                encoding="utf-8")
                temp.replace(self.path)

    def start(self, title: str, chapters: list[tuple[str, str]], settings: dict | None = None) -> dict:
        job_id = uuid.uuid4().hex[:10]
        job = {"id": job_id, "title": title, "status": "running", "done": 0, "total": len(chapters),
               "chapters": [t for t, _ in chapters], "items": [], "error": None, "started": time.time()}
        self.jobs[job_id] = job
        self.work[job_id] = {"chapters": [list(c) for c in chapters], "settings": settings or {}}
        self._save()
        self._run(job)
        return job

    def _run(self, job: dict) -> None:
        work = self.work[job["id"]]

        def run():
            for chapter_title, text in work["chapters"][job["done"]:]:
                if job["status"] != "running":  # cancelled; stops after the chapter in progress
                    break
                for attempt in (1, 2):  # one more try: a chapter can fail on something passing
                    try:
                        meta = self.read(job["title"], work["settings"], chapter_title, text, job["id"])
                        break
                    except Exception as e:  # report it in the job; the server keeps running
                        if attempt == 2:
                            job["status"], job["error"] = "failed", f"{chapter_title}: {e}"
                            self._save()
                            return
                job["items"].append(meta["id"])
                job["done"] += 1
                self._save()
            if job["status"] == "running":
                job["status"] = "done"
            self._save()

        threading.Thread(target=run, daemon=True, name=f"document-{job['id']}").start()

    def get(self, job_id: str) -> dict:
        return self.jobs[job_id]

    def cancel(self, job_id: str) -> dict:
        job = self.jobs[job_id]
        if job["status"] == "running":
            job["status"] = "cancelled"  # stops after the chapter in progress
            self._save()
        return job

    def resume(self, job_id: str) -> dict:
        """Goes on from the first chapter not read yet. Raises ValueError if the job can't."""
        job = self.jobs[job_id]
        if job["status"] not in self.RESUMABLE or job_id not in self.work or job["done"] >= job["total"]:
            raise ValueError("That job can't be continued.")
        job["status"], job["error"] = "running", None
        self._save()
        self._run(job)
        return job


def zip_chapters(paths: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # MP3 doesn't compress further
        for name, data in paths:
            z.writestr(name, data)
    return buf.getvalue()
