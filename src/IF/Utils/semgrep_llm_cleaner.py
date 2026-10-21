import re
import os
import tempfile
import textwrap
from typing import List, Optional, Tuple

# Common separators and markers LLMs sometimes inject
SEPARATOR_PATTERNS = [
    r"<\|file_separator\|>",
    r"--8<--",
    r"={3,}",               # lines with === or more
    r"-{3,}",               # lines like -----
    r"\[-- file:.*?--\]",
]

# Regex to capture fenced code blocks:
# - group 1: fence info (e.g. "python", "python filename.py", "filename.py")
# - group 2: code content
FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Regex for header-style filename hints like:
#   # file: foo.py
#   // file: foo.py
#   --- file: foo.py ---
HEADER_FILE_RE = re.compile(r"^(?:[#/\\*-]{0,3}\s*)file[:=]\s*(?P<fn>[^\n\r]+)",
                            re.IGNORECASE | re.MULTILINE)

# Remove control characters except common whitespace (\t, \n, \r)
NON_PRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")

LEADING_NOISE_LINE_RE = re.compile(
    r"""^(
        (\s*[\.\-_\*]{1,}\s*$)         |  # lines that are only punctuation like ".", "...", "----"
        (\s*[\w\-.]+\.json\s*$)       |  # lone "something.json" lines
        (\s*[\w\-.]+\.py\s*$)         |  # lone "something.py" lines
        (\s*<\|file_separator\|>\s*$) |  # explicit LLM separator token lines
        (\s*```(?:python)?\s*$)       |  # lone fence start/end
        (\s*file[:=]\s*.+$)              # "file: name" header
    )""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

def _strip_leading_noise(snippet: str) -> str:
    snippet = snippet.replace("\r\n", "\n").replace("\r", "\n")
    snippet = snippet.lstrip('\ufeff')  # remove BOM
    lines = snippet.splitlines()

    # Remove leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)

    max_drop = min(10, len(lines))
    dropped = 0
    while dropped < max_drop and lines:
        if LEADING_NOISE_LINE_RE.match(lines[0]):
            lines.pop(0)
            dropped += 1
            while lines and not lines[0].strip():
                lines.pop(0)
        else:
            break

    cleaned = "\n".join(lines)
    return cleaned.strip("\n") + ("\n" if cleaned and not cleaned.endswith("\n") else "")

def _sanitize_text(s: str) -> str:
    """Normalize line endings, remove nonprintable chars, collapse separators to newline."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = NON_PRINTABLE_RE.sub("", s)
    for pat in SEPARATOR_PATTERNS:
        s = re.sub(pat, "\n", s)
    return s

def _guess_filename_from_fence_info(fence_info: str) -> Optional[str]:
    """
    Try to extract filename from the fence info token.
    Examples of fence_info:
      "python"
      "python filename.py"
      "filename.py"
      "text filename=foo.py"
    """
    if not fence_info:
        return None
    # look for something that looks like a filename with extension
    m = re.search(r"([\w\-.]+\.py)\b", fence_info)
    if m:
        return m.group(1)
    # sometimes language is omitted and fence_info is just the filename
    token = fence_info.strip()
    if token.endswith(".py"):
        return token
    return None

def _guess_filename_from_snippet(snippet: str) -> Optional[str]:
    """Search common 'file:' headers in snippet to find a filename."""
    m = HEADER_FILE_RE.search(snippet)
    if m:
        fn = m.group("fn").strip()
        # sanitize filename (remove enclosing quotes or trailing punctuation)
        fn = fn.strip(" \"'`")
        # take base name only
        fn = os.path.basename(fn)
        if fn.endswith(".py"):
            return fn
    return None

def _unique_filename(directory: str, base: str, ext: str = ".py") -> str:
    """Return a path that does not clash with an existing file in directory."""
    base = re.sub(r"[^\w\-.]", "_", base) or "snippet"
    candidate = f"{base}{ext}"
    i = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return os.path.join(directory, candidate)

def _extract_snippets(cleaned_text: str, split_on: Optional[str] = None) -> List[Tuple[Optional[str], str]]:
    """
    Return list of tuples (optional_filename_hint, code_snippet).
    Extraction order:
      1. If fenced code blocks exist -> each fence produces a snippet; attempt to extract filename from fence info.
      2. Else if split_on token provided -> split on that
      3. Else if separators found -> split heuristically on separator patterns
      4. Else return the whole cleaned_text as single snippet
    """
    snippets: List[Tuple[Optional[str], str]] = []

    # 1) fenced code blocks
    fences = FENCE_RE.findall(cleaned_text)
    if fences:
        for fence_info, code in fences:
            filename_hint = _guess_filename_from_fence_info(fence_info)
            snippets.append((filename_hint, code.strip("\n")))
        return snippets

    # 2) split_on token (explicit user-provided)
    if split_on:
        parts = [p.strip() for p in cleaned_text.split(split_on) if p.strip()]
        for p in parts:
            fn = _guess_filename_from_snippet(p)
            snippets.append((fn, p))
        return snippets

    # 3) heuristic split by separator patterns
    for pat in SEPARATOR_PATTERNS:
        if re.search(pat, cleaned_text):
            parts = [p.strip() for p in re.split(pat, cleaned_text) if p.strip()]
            for p in parts:
                fn = _guess_filename_from_snippet(p)
                snippets.append((fn, p))
            return snippets

    # 4) fallback: whole text
    fn = _guess_filename_from_snippet(cleaned_text)
    snippets.append((fn, cleaned_text.strip("\n")))
    return snippets

def clean_and_save(llm_text: str,
                   output_dir: Optional[str] = None,
                   default_name_prefix: str = "llm_code",
                   split_on: Optional[str] = None) -> List[str]:
    """
    Clean LLM output and save one or more .py files to output_dir (or a tempdir).
    Returns the list of file paths written.

    - llm_text: raw LLM output string
    - output_dir: directory to write files to (created if necessary). If None, a temp dir is used.
    - default_name_prefix: prefix for generated filenames when no hint is present.
    - split_on: optional explicit splitter token (e.g. "<|file_separator|>")
    """
    cleaned = _sanitize_text(llm_text)

    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="semgrep_llm_out_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    snippets = _extract_snippets(cleaned, split_on=split_on)

    written_files: List[str] = []
    for idx, (hint, snippet) in enumerate(snippets, start=1):
        # If snippet looks like it starts with a language marker, remove it
        snippet = re.sub(r"^```[^\n]*\n?", "", snippet)
        snippet = re.sub(r"\n?```$", "", snippet)

        # Remove "file: foo.py" header lines so they don't end up in the code
        snippet = re.sub(r"^(?:[#/\\*-]{0,3}\s*)file[:=]\s*.*$", "", snippet, flags=re.IGNORECASE | re.MULTILINE).strip("\n")

        # Determine filename
        if hint:
            base = hint
        else:
            # try to discover a sensible base from first non-empty line (but sanitize)
            first_line = ""
            for ln in snippet.splitlines():
                if ln.strip():
                    first_line = ln.strip()
                    break
            # pick prefix from first token or default prefix
            tok = re.split(r"\s+", first_line)[0] if first_line else default_name_prefix
            # sanitize token
            tok = re.sub(r"[^\w\-.]", "_", tok)[:30]  # limit length
            base = f"{default_name_prefix}_{tok}_{idx}"

        path = _unique_filename(output_dir, base)
        # ensure .py extension
        if not path.endswith(".py"):
            path = f"{path}.py"

        # final clean: strip leading/trailing blank lines
        #snippet = snippet.strip("\n") + "\n"
        
        # --- clean leading noise before writing ---
        snippet = _strip_leading_noise(snippet)

        # final clean: strip trailing blanks only
        snippet = snippet.rstrip() + "\n"

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(snippet)

        written_files.append(path)

    return written_files