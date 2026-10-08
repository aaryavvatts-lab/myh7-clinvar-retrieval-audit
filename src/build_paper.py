"""Build the v6 paper deliverables from paper/paper_src.md and paper/references.md.

Outputs (in paper/):  paper_v6.md (full Markdown, pandoc-style pipe tables, image links),
                      paper_v6.tex (LaTeX), and, if tectonic is available, MYH7_paper_v6.pdf.
A tiny Markdown subset is converted by hand because pandoc is not installed:
  headings (##, ###), paragraphs, bullet and numbered lists, pipe tables followed by 'Table: caption',
  images '![caption](path){width=0.9}', display math '$$ ... $$', inline $math$, *italic*, **bold**, bare URLs.
Run from v6/:  ../.venv/bin/python -m src.build_paper
"""
from __future__ import annotations
import re, shutil, subprocess
from pathlib import Path

P = Path("paper")
import os, shutil
TECTONIC = os.environ.get("TECTONIC") or shutil.which("tectonic") or str(Path.home() / ".local" / "bin" / "tectonic")

SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def esc(t: str) -> str:
    return "".join(SPECIAL.get(c, c) for c in t)


def inline(t: str) -> str:
    """Markdown inline -> LaTeX. Math is protected; URLs become \\url{}."""
    store = []

    def keep(m):
        store.append(m.group(0)); return f"\x00{len(store)-1}\x00"
    t = re.sub(r"\$[^$]+\$", keep, t)
    def keepc(m):
        store.append(r"\texttt{\seqsplit{" + esc(m.group(1)) + "}}"); return f"\x00{len(store)-1}\x00"
    t = re.sub(r"`([^`]+)`", keepc, t)
    urls = []

    def keepu(m):
        urls.append(m.group(0)); return f"\x01{len(urls)-1}\x01"
    t = re.sub(r"https?://[^\s)]+", keepu, t)
    # bold / italic markers -> placeholders before escaping
    t = re.sub(r"\*\*(.+?)\*\*", lambda m: "\x02" + m.group(1) + "\x03", t)
    t = re.sub(r"\*(.+?)\*", lambda m: "\x04" + m.group(1) + "\x05", t)
    t = esc(t)
    t = re.sub(r"(?<![\w$\x00])-(?=\d)", r"$-$", t)          # minus sign before numbers
    t = t.replace("\x02", r"\textbf{").replace("\x03", "}").replace("\x04", r"\emph{").replace("\x05", "}")
    t = re.sub(r"\x01(\d+)\x01", lambda m: r"\url{" + urls[int(m.group(1))] + "}", t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], t)
    return t


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_tex(rows, caption):
    head = split_row(rows[0]); align = split_row(rows[1]); body = [split_row(r) for r in rows[2:]]
    n = len(head)
    w1 = 1.7 if n > 3 else 1.4
    wo = (n - w1) / (n - 1)
    spec = []
    for i, a in enumerate(align):
        if i == 0:
            spec.append(rf">{{\hsize={w1:.3f}\hsize\raggedright\arraybackslash}}X")
        else:
            al = r"\raggedleft" if a.endswith(":") else r"\raggedright"
            spec.append(rf">{{\hsize={wo:.3f}\hsize{al}\arraybackslash}}X")
    size = r"\scriptsize" if n >= 6 else (r"\footnotesize" if n >= 4 else r"\small")
    out = [r"\begin{table}[htbp]", r"\centering", size, r"\caption{" + inline(caption) + "}",
           r"\begin{tabularx}{\textwidth}{" + " ".join(spec) + "}", r"\toprule",
           " & ".join(inline(h) for h in head) + r" \\", r"\midrule"]
    for r in body:
        out.append(" & ".join(inline(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(out)


def figure_tex(caption, path, width):
    return "\n".join([r"\begin{figure}[htbp]", r"\centering",
                      rf"\includegraphics[width={width}\textwidth]{{{path}}}",
                      r"\caption{" + inline(caption) + "}", r"\end{figure}"])


def convert(src: str, refs: str):
    fm, body = src.split("---\n", 2)[1:]
    meta = dict(re.findall(r'^(\w+):\s*"(.*)"\s*$', fm, flags=re.M))
    lines = body.split("\n")
    tex, i = [], 0
    in_abstract = False
    ref_mode = False
    para = []

    def flush():
        nonlocal para
        if para:
            text = " ".join(para).strip()
            if ref_mode:
                tex.append(r"\hangindent=1.5em\hangafter=1 " + inline(text) + r"\par\vspace{2pt}")
            else:
                tex.append(inline(text) + "\n")
            para = []
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            flush(); i += 1; continue
        m = re.match(r"^(#{2,3}) (.*)$", ln)
        if m:
            flush()
            if in_abstract:
                tex.append(r"\end{abstract}"); in_abstract = False
            lvl, title = len(m.group(1)), m.group(2)
            num = re.match(r"^(\d+(?:\.\d+)*)\s+(.*)$", title)
            if title == "Abstract":
                tex.append(r"\begin{abstract}"); in_abstract = True
            elif title == "References":
                ref_mode = True
                tex.append(r"\section*{References}" + "\n" + r"{\small\setlength{\parindent}{0pt}")
            elif num:
                tex.append((r"\section{" if lvl == 2 else r"\subsection{") + inline(num.group(2)) + "}")
            else:
                tex.append((r"\section*{" if lvl == 2 else r"\subsection*{") + inline(title) + "}")
            i += 1; continue
        if ln.startswith("!["):
            flush()
            m = re.match(r"^!\[(.*)\]\((.*?)\)(?:\{width=([\d.]+)\})?\s*$", ln)
            cap = re.sub(r"^\*\*Figure \d+\.\*\*\s*", "", m.group(1))
            tex.append(figure_tex(cap, m.group(2), m.group(3) or "0.95")); i += 1; continue
        if ln.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            cap = re.sub(r"^Table:\s*\*\*Table \d+\.\*\*\s*", "", lines[i]); i += 1
            tex.append(table_tex(rows, cap)); continue
        if ln.startswith("$$"):
            flush(); tex.append(r"\[" + ln.strip().strip("$").strip() + r"\]"); i += 1; continue
        if re.match(r"^(- |\d+\. )", ln):
            flush()
            ordered = bool(re.match(r"^\d+\. ", ln))
            items = []
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i]):
                item = re.sub(r"^(- |\d+\. )", "", lines[i]); i += 1
                while i < len(lines) and lines[i].strip() and not re.match(r"^(- |\d+\. |#|\||!\[)", lines[i]):
                    item += " " + lines[i].strip(); i += 1
                items.append(item)
            env = "enumerate" if ordered else "itemize"
            tex.append(rf"\begin{{{env}}}[nosep]" + "".join(r"\item " + inline(it) + "\n" for it in items) + rf"\end{{{env}}}")
            continue
        para.append(ln.strip()); i += 1
    flush()
    if in_abstract:
        tex.append(r"\end{abstract}")
    if ref_mode:
        tex.append("}")
    pre = r"""\documentclass[11pt]{article}
\usepackage[letterpaper,margin=1in]{geometry}
\usepackage{fontspec}
\setmainfont[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]{texgyrepagella}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{enumitem}
\usepackage{placeins}
\usepackage[labelfont=bf,labelsep=period,font=small,justification=justified,singlelinecheck=false]{caption}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\usepackage{seqsplit}
\linespread{1.08}
\setlength{\parindent}{1.2em}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\abstractname}{Abstract}
\makeatletter
\renewenvironment{abstract}{\small\begin{quote}\textbf{\abstractname.}\ }{\end{quote}}
\makeatother
\title{\textbf{""" + inline(meta["title"]) + r"""}}
\author{""" + inline(meta["author"]) + r"""\\ \small """ + inline(meta["affiliation"]) + r"""}
\date{""" + inline(meta["date"]) + r"""}
\begin{document}
\maketitle
"""
    post = "\n\\end{document}\n"
    return pre + "\n".join(tex) + post, meta


def main():
    src = (P / "paper_src.md").read_text()
    refs = (P / "references.md").read_text().strip()
    full_md = src.replace("{{REFERENCES}}", refs)
    # Markdown deliverable: add a title block and use relative image links
    (P / "paper_v6.md").write_text(full_md)
    tex, meta = convert(src.replace("{{REFERENCES}}", refs), refs)
    (P / "paper_v6.tex").write_text(tex)
    r = subprocess.run([TECTONIC, "--keep-logs", "paper_v6.tex"], cwd=P, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-3000:])
    if (P / "paper_v6.pdf").exists():
        shutil.copy(P / "paper_v6.pdf", P / "MYH7_paper_v6.pdf")


if __name__ == "__main__":
    main()
