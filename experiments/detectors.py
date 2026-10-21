"""
Run three vulnerability detectors (Bandit, Semgrep, AST taint tracker) on every
code snippet in a results pickle, and store a binary flag per detector.

The pickle is expected to look like  data[key][code_snippet] = {...}.
For each snippet, the following entries are (over)written:
    data[key][code_snippet]["Bandit"]            = [1.0 or 0.0]
    data[key][code_snippet]["Semgrep"]           = [1.0 or 0.0]
    data[key][code_snippet]["AST Taint Tracker"] = [1.0 or 0.0]
where 1.0 means the detector reported at least one finding.

Usage:
    python detectors.py --datasetName results.pkl [--datasetDir /path/to/dir/]

The detectors can also be run directly on one piece of code, without a pickle.
Nothing is saved; the number of findings per detector is printed:
    python detectors.py --code 'import os; os.system(input())'
    python detectors.py --codeFile snippet.py
    cat snippet.py | python detectors.py --codeFile -
"""

import argparse
import ast
import json
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

from IF import RESULTS_PATH, SEMGREP_INSTRUCTIONS_PATH
from IF.Utils.semgrep_llm_cleaner import clean_and_save

SEMGREP_RULES_DIR = str(SEMGREP_INSTRUCTIONS_PATH)
DEFAULT_DATASET_DIR = str(RESULTS_PATH)


# =============================================================================
# Semgrep
# =============================================================================

def naive_semgrep(code: str) -> int:
    """Number of Semgrep findings, using the rules in SEMGREP_RULES_DIR.

    `clean_and_save` writes the snippet to temporary .py file(s); only the
    first file is scanned (as in the original implementation).
    """
    for code_file in clean_and_save(code):
        proc = subprocess.run(
            ["semgrep", "--config", SEMGREP_RULES_DIR, "--json", code_file],
            capture_output=True, text=True, timeout=180,
        )
        os.remove(code_file)
        return len(json.loads(proc.stdout)["results"])


# =============================================================================
# Bandit
# =============================================================================

def run_bandit(code: str, idx: int = -1) -> int:
    """Number of issues Bandit reports for the snippet."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False,
                                     prefix=f"b_{idx}_") as f:
        f.write(code)
        tmp_path = f.name

    proc = subprocess.run(
        ["bandit", "-r", tmp_path, "-f", "json"],
        capture_output=True, text=True, timeout=30,
    )
    return len(json.loads(proc.stdout).get("results", []))


# =============================================================================
# AST taint tracker
#
# Tracks values coming from Flask's `request` object ("sources") through
# assignments, and records a "flow" whenever a tainted value reaches a
# dangerous function ("sink"), a response header, or an HTML response.
# =============================================================================

# request.<attr> attributes that are user-controlled
_ATTR_SOURCES = {
    ("request", "args"):    "UserControlled",
    ("request", "form"):    "UserControlled",
    ("request", "json"):    "UserControlled",
    ("request", "data"):    "UserControlled",
    ("request", "values"):  "UserControlled",
    ("request", "files"):   "UserControlled",
    ("request", "cookies"): "Cookies",
    ("request", "headers"): "UserControlled",
}

# Calls that return user-controlled data
_CALL_SOURCES = {
    "request.get_json", "request.get_data",
    "request.args.get", "request.form.get",
    "request.values.get", "request.cookies.get", "request.headers.get",
}

# Sink function (full dotted name or last component) -> (sink_kind, severity, rule_name)
_SINKS: Dict[str, Tuple[str, str, str]] = {
    # SQL
    "execute":                 ("SQL", "HIGH", "SQLInjection"),
    "executemany":             ("SQL", "HIGH", "SQLInjection"),
    "executescript":           ("SQL", "HIGH", "SQLInjection"),
    "raw":                     ("SQL", "HIGH", "SQLInjection"),
    # Command execution / RCE
    "os.system":               ("RCE", "CRITICAL", "RemoteCodeExecution"),
    "os.popen":                ("RCE", "CRITICAL", "RemoteCodeExecution"),
    "subprocess.run":          ("RCE", "CRITICAL", "RemoteCodeExecution"),
    "subprocess.call":         ("RCE", "CRITICAL", "RemoteCodeExecution"),
    "subprocess.Popen":        ("RCE", "CRITICAL", "RemoteCodeExecution"),
    "subprocess.check_output": ("RCE", "CRITICAL", "RemoteCodeExecution"),
    "subprocess.check_call":   ("RCE", "CRITICAL", "RemoteCodeExecution"),
    # Code injection
    "eval":                    ("CodeExec", "CRITICAL", "CodeInjection"),
    "exec":                    ("CodeExec", "CRITICAL", "CodeInjection"),
    "compile":                 ("CodeExec", "HIGH",     "CodeInjection"),
    # Filesystem / path traversal
    "open":                    ("FileSystem", "HIGH",   "PathTraversal"),
    "send_file":               ("FileSystem", "HIGH",   "PathTraversal"),
    "send_from_directory":     ("FileSystem", "HIGH",   "PathTraversal"),
    "os.path.abspath":         ("FileSystem", "MEDIUM", "PathTraversal"),
    "os.path.join":            ("FileSystem", "HIGH",   "PathTraversal"),
    "os.path.realpath":        ("FileSystem", "MEDIUM", "PathTraversal"),
    "os.path.dirname":         ("FileSystem", "MEDIUM", "PathTraversal"),
    "os.listdir":              ("FileSystem", "HIGH",   "PathTraversal"),
    "os.walk":                 ("FileSystem", "HIGH",   "PathTraversal"),
    "os.scandir":              ("FileSystem", "HIGH",   "PathTraversal"),
    "os.remove":               ("FileSystem", "HIGH",   "PathTraversal"),
    "os.unlink":               ("FileSystem", "HIGH",   "PathTraversal"),
    "os.rmdir":                ("FileSystem", "HIGH",   "PathTraversal"),
    "os.rename":               ("FileSystem", "HIGH",   "PathTraversal"),
    "os.chmod":                ("FileSystem", "HIGH",   "PathTraversal"),
    "shutil.copy":             ("FileSystem", "HIGH",   "PathTraversal"),
    "shutil.move":             ("FileSystem", "HIGH",   "PathTraversal"),
    "shutil.rmtree":           ("FileSystem", "HIGH",   "PathTraversal"),
    # Deserialization
    "pickle.load":             ("Deserialize", "CRITICAL", "InsecureDeserialization"),
    "pickle.loads":            ("Deserialize", "CRITICAL", "InsecureDeserialization"),
    "yaml.load":               ("Deserialize", "HIGH",     "InsecureDeserialization"),
    "marshal.load":            ("Deserialize", "HIGH",     "InsecureDeserialization"),
    "marshal.loads":           ("Deserialize", "HIGH",     "InsecureDeserialization"),
    "jsonpickle.decode":       ("Deserialize", "HIGH",     "InsecureDeserialization"),
    "dill.loads":              ("Deserialize", "HIGH",     "InsecureDeserialization"),
    # SSRF
    "requests.get":            ("SSRF", "HIGH", "SSRF"),
    "requests.post":           ("SSRF", "HIGH", "SSRF"),
    "requests.put":            ("SSRF", "HIGH", "SSRF"),
    "requests.delete":         ("SSRF", "HIGH", "SSRF"),
    "requests.request":        ("SSRF", "HIGH", "SSRF"),
    "urllib.request.urlopen":  ("SSRF", "HIGH", "SSRF"),
    "urlopen":                 ("SSRF", "HIGH", "SSRF"),
    "pdfkit.from_url":         ("SSRF", "HIGH", "SSRF"),
    # XSS / template injection
    "render_template_string":  ("XSS",  "HIGH",   "XSS"),
    "make_response":           ("XSS",  "MEDIUM", "XSS"),
    "Markup":                  ("XSS",  "HIGH",   "XSS"),
    "Template":                ("SSTI", "HIGH",   "TemplateInjection"),
    "env.from_string":         ("SSTI", "HIGH",   "TemplateInjection"),
    # Redirect
    "redirect":                ("OpenRedirect", "HIGH", "OpenRedirect"),
    # File upload
    "file.save":               ("FileUpload", "HIGH", "UnrestrictedFileUpload"),
    "save":                    ("FileUpload", "HIGH", "UnrestrictedFileUpload"),
    # XXE
    "ET.fromstring":           ("XXE", "HIGH", "XXE"),
    "ET.parse":                ("XXE", "HIGH", "XXE"),
    "ET.XML":                  ("XXE", "HIGH", "XXE"),
    "etree.fromstring":        ("XXE", "HIGH", "XXE"),
    "etree.parse":             ("XXE", "HIGH", "XXE"),
    "lxml.etree.fromstring":   ("XXE", "HIGH", "XXE"),
    "lxml.etree.parse":        ("XXE", "HIGH", "XXE"),
    # Log injection
    "logging.info":            ("LogInjection", "MEDIUM", "LogInjection"),
    "logging.warning":         ("LogInjection", "MEDIUM", "LogInjection"),
    "logging.error":           ("LogInjection", "MEDIUM", "LogInjection"),
    "logging.debug":           ("LogInjection", "MEDIUM", "LogInjection"),
    "logging.critical":        ("LogInjection", "MEDIUM", "LogInjection"),
    # NoSQL (MongoDB / Redis methods)
    "find":                    ("NoSQL", "HIGH",   "NoSQLInjection"),
    "find_one":                ("NoSQL", "HIGH",   "NoSQLInjection"),
    "update_one":              ("NoSQL", "HIGH",   "NoSQLInjection"),
    "update_many":             ("NoSQL", "HIGH",   "NoSQLInjection"),
    "delete_one":              ("NoSQL", "HIGH",   "NoSQLInjection"),
    "delete_many":             ("NoSQL", "HIGH",   "NoSQLInjection"),
    "insert_one":              ("NoSQL", "MEDIUM", "NoSQLInjection"),
    "insert_many":             ("NoSQL", "MEDIUM", "NoSQLInjection"),
    "aggregate":               ("NoSQL", "HIGH",   "NoSQLInjection"),
    "keys":                    ("NoSQL", "HIGH",   "NoSQLInjection"),
    # Dynamic import
    "importlib.import_module": ("ImportInjection", "HIGH", "DynamicImport"),
    "__import__":              ("ImportInjection", "HIGH", "DynamicImport"),
    # LDAP
    "search_s":                ("LDAP", "HIGH", "LDAPInjection"),
    "search":                  ("LDAP", "HIGH", "LDAPInjection"),
}

# Variable names that, when returned tainted, are treated as an HTML response
_HTML_RESPONSE_NAMES = {"response", "html", "content", "page", "body", "output", "result", "rendered"}

# SQL sinks whose first argument, if a plain string literal, means a parameterised query
_PARAMETERISED_SQL_SINKS = {"execute", "executemany"}

Taint = Tuple[str, int]  # (source_kind, source_line)


def _dotted_name(node) -> str:
    """'a.b.c' for Name/Attribute chains, '' otherwise."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    return ""


def _attr_source(dotted: str, node) -> Optional[Taint]:
    """Taint if `dotted` ends in a user-controlled request attribute, e.g. 'request.args'."""
    parts = dotted.split(".")
    if len(parts) >= 2 and (parts[-2], parts[-1]) in _ATTR_SOURCES:
        return _ATTR_SOURCES[(parts[-2], parts[-1])], getattr(node, "lineno", 0)
    return None


def _fstring_has_html(node) -> bool:
    """True if an f-string's literal parts contain '<' or '>'."""
    return isinstance(node, ast.JoinedStr) and any(
        isinstance(v, ast.Constant) and isinstance(v.value, str) and ("<" in v.value or ">" in v.value)
        for v in node.values
    )


class _TaintVisitor(ast.NodeVisitor):
    def __init__(self):
        self.tainted: Dict[str, Taint] = {}   # variable name -> taint
        self.flows: List[dict] = []           # detected source -> sink flows

    # ---- helpers ------------------------------------------------------------

    def _first_taint(self, nodes) -> Optional[Taint]:
        for n in nodes:
            t = self._taint(n)
            if t:
                return t
        return None

    def _add_flow(self, rule_name, taint, sink_fn, sink_kind, node, severity="HIGH"):
        self.flows.append({
            "rule_name":   rule_name,
            "source_kind": taint[0],
            "source_line": taint[1],
            "sink_fn":     sink_fn,
            "sink_kind":   sink_kind,
            "sink_line":   getattr(node, "lineno", 0),
            "severity":    severity,
        })

    def _taint(self, node) -> Optional[Taint]:
        """Taint carried by an expression, or None if it is clean."""
        if isinstance(node, ast.Name):
            return self.tainted.get(node.id)
        if isinstance(node, ast.Subscript):                       # request.args["x"]
            return _attr_source(_dotted_name(node.value), node) or self._taint(node.value)
        if isinstance(node, ast.Attribute):                       # request.json
            return _attr_source(_dotted_name(node), node) or self._taint(node.value)
        if isinstance(node, ast.Call):                            # request.args.get("x"), f(tainted)
            if _dotted_name(node.func) in _CALL_SOURCES:
                return "UserControlled", getattr(node, "lineno", 0)
            receiver = [node.func.value] if isinstance(node.func, ast.Attribute) else []
            return self._first_taint(receiver + node.args + [kw.value for kw in node.keywords])
        if isinstance(node, ast.BinOp):                           # "a" + tainted
            return self._taint(node.left) or self._taint(node.right)
        if isinstance(node, ast.JoinedStr):                       # f"...{tainted}..."
            return self._first_taint(node.values)
        if isinstance(node, ast.FormattedValue):
            return self._taint(node.value)
        if isinstance(node, (ast.List, ast.Tuple)):
            return self._first_taint(node.elts)
        if isinstance(node, ast.IfExp):
            return self._taint(node.body) or self._taint(node.orelse)
        if isinstance(node, ast.Dict):
            return self._first_taint(node.values)
        return None

    def _propagate(self, target, value_node):
        """Assigning to a variable taints it, or clears its taint."""
        if not isinstance(target, ast.Name):
            return
        t = self._taint(value_node)
        if t:
            self.tainted[target.id] = t
        else:
            self.tainted.pop(target.id, None)

    # ---- assignments ----------------------------------------------------------

    def visit_Assign(self, node):
        for target in node.targets:
            self._propagate(target, node.value)

            # Header injection: response.headers[key] = tainted
            if isinstance(target, ast.Subscript):
                obj_name = _dotted_name(target.value)
                t = self._taint(node.value) if "headers" in obj_name else None
                if t:
                    self._add_flow("HeaderInjection", t, obj_name + "[key]", "HttpHeader", node)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            t = self._taint(node.value)
            if t:
                self.tainted[node.target.id] = t
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value:
            self._propagate(node.target, node.value)
        self.generic_visit(node)

    # ---- XSS via return ---------------------------------------------------------

    def visit_Return(self, node):
        value = node.value

        # return f"<html>{tainted}</html>"
        if isinstance(value, ast.JoinedStr) and _fstring_has_html(value):
            t = self._taint(value)
            if t:
                self._add_flow("XSS", t, 'return f"<...>{tainted}..."', "HtmlResponse", node)

        # return html   (a tainted variable with an HTML-like name)
        elif isinstance(value, ast.Name) and value.id in self.tainted:
            if value.id.lower() in _HTML_RESPONSE_NAMES:
                self._add_flow("XSS", self.tainted[value.id], f"return {value.id}  (tainted)",
                               "HtmlResponse", node)

        # return "<html>" + tainted
        elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            t = self._taint(value)
            left = value.left
            if t and isinstance(left, ast.Constant) and isinstance(left.value, str) and "<" in left.value:
                self._add_flow("XSS", t, 'return "<html>" + tainted', "HtmlResponse", node)

        self.generic_visit(node)

    # ---- calls to dangerous sinks ------------------------------------------------

    def visit_Call(self, node):
        fn = _dotted_name(node.func)
        fn_leaf = fn.split(".")[-1]

        # cursor.execute("... ? ...", params) is a parameterised query: not a sink
        first_arg = node.args[0] if node.args else None
        if (fn_leaf in _PARAMETERISED_SQL_SINKS
                and isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)):
            self.generic_visit(node)
            return

        # Tainted positional/keyword argument, or a tainted *args spread
        tainted_arg = (
            self._first_taint(node.args)
            or self._first_taint(kw.value for kw in node.keywords)
            or self._first_taint(a.value for a in node.args if isinstance(a, ast.Starred))
        )

        if tainted_arg:
            for sink in ([fn, fn_leaf] if "." in fn else [fn]):
                if sink in _SINKS:
                    sink_kind, severity, rule_name = _SINKS[sink]
                    self._add_flow(rule_name, tainted_arg, fn, sink_kind, node, severity)
                    break
        self.generic_visit(node)


def run_ast_taint_tracker(code: str) -> int:
    """Number of distinct (rule, line) taint flows; 0 if the code doesn't parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    try:
        visitor = _TaintVisitor()
        visitor.visit(tree)
    except Exception:
        return 0
    return len({(f["rule_name"], f["sink_line"]) for f in visitor.flows})


# =============================================================================
# Main
# =============================================================================

def flag(num_findings) -> List[float]:
    """[1.0] if the detector found anything, else [0.0]."""
    return [1.0 if float(num_findings) > 0 else 0.0]


def run_detectors(data: dict) -> dict:
    for group in data.values():
        for code, entry in group.items():
            entry["Bandit"] = flag(run_bandit(code))
            entry["Semgrep"] = flag(naive_semgrep(code))
            entry["AST Taint Tracker"] = flag(run_ast_taint_tracker(code))

            # Drop stale keys from older versions of this script
            entry.pop("Custom  Semgrep", None)
            entry.pop("Naive  Semgrep", None)
    return data


DETECTORS = {
    "Bandit": run_bandit,
    "Semgrep": naive_semgrep,
    "AST Taint Tracker": run_ast_taint_tracker,
}


def run_on_code(code: str) -> dict:
    """Run every detector on one snippet and print the number of findings.

    Unlike dataset mode, a detector that fails does not stop the others: its
    error is printed instead of a count.
    """
    results = {}
    for name, detector in DETECTORS.items():
        try:
            count = detector(code)
            if count is None:
                results[name] = "no result (clean_and_save produced no file to scan)"
            else:
                results[name] = int(count)
        except Exception as e:
            results[name] = f"error: {type(e).__name__}: {e}"

    for name, value in results.items():
        if isinstance(value, int):
            print(f"{name:18s} {value} finding(s) -> flag {1.0 if value > 0 else 0.0}")
        else:
            print(f"{name:18s} {value}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run detectors on a dataset pickle, or directly on one piece of code.")
    parser.add_argument("--datasetName", type=str, default=None,
                        help="Dataset (pickle) file name.")
    parser.add_argument("--datasetDir", type=str, default=DEFAULT_DATASET_DIR,
                        help="Directory containing the dataset.")
    parser.add_argument("--code", type=str, default=None,
                        help="Code to scan, given directly as text (use quotes).")
    parser.add_argument("--codeFile", type=str, default=None,
                        help="Path of a file containing the code to scan; use - for stdin.")
    args = parser.parse_args()

    chosen = [a for a in (args.datasetName, args.code, args.codeFile) if a is not None]
    if len(chosen) != 1:
        parser.error("give exactly one of --datasetName, --code or --codeFile")

    # ---- direct mode: scan one snippet and print the results ----
    if args.code is not None or args.codeFile is not None:
        if args.code is not None:
            code = args.code
        elif args.codeFile == "-":
            code = sys.stdin.read()
        else:
            with open(args.codeFile) as f:
                code = f.read()
        run_on_code(code)
        return

    # ---- dataset mode: scan every snippet in the pickle and save the flags ----
    path = os.path.join(args.datasetDir, args.datasetName)

    with open(path, "rb") as f:
        data = pickle.load(f)

    data = run_detectors(data)

    with open(path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()