"""Writing-quality and consistency checks for paper/paper_v6.md (run from v6/)."""
import re, sys
from pathlib import Path
md = Path("paper/paper_v6.md").read_text()
body, refs = md.split("## References")
print("em/en dashes:", len(re.findall("[—–]", md)), "| curly quotes:", len(re.findall("[‘’“”]", md)))
print("straight double quotes in body:", body.count('"') - body.split("---", 2)[1].count('"'))
banned = ["delve", "crucial", "robust", "underscore", "pivotal", "landscape", "showcase", "serves as", "stands as", "moreover",
          "furthermore", "not just", "in conclusion", "novel", "comprehensive", "leverage", "foster", "tapestry", "multifaceted", "nuanced", "significant"]
for b in banned:
    for part, name in ((body, "body"), (refs, "refs")):
        n = len(re.findall(b, part, flags=re.I))
        if n: print(f"banned word '{b}': {n} in {name}")
# sentence length (prose only: drop tables, figure captions, math, headings, code)
prose = re.sub(r"^\|.*$|^Table:.*$|^!\[.*$|^\$\$.*$|^#.*$|^---.*$|^(title|author|affiliation|date):.*$", "", body, flags=re.M)
prose = re.sub(r"\$[^$]+\$", "X", prose); prose = re.sub(r"`[^`]+`", "H", prose)
sents = [s for s in re.split(r"(?<=[.?!])\s+(?=[A-Z0-9*])", re.sub(r"\s+", " ", prose)) if len(s.split()) > 2]
words = [len(s.split()) for s in sents]
print(f"sentences {len(sents)}, mean length {sum(words)/len(words):.1f} words, share over 30 words {sum(w>30 for w in words)/len(words):.1%}")
# abstract words
ab = re.search(r"## Abstract\s+(.*?)\n\n\*\*Keywords", body, flags=re.S).group(1)
print("abstract words:", len(ab.split()), "| contains citation parens with year:", bool(re.search(r"\(\w[^)]*\d{4}\)", ab)))
# figures / tables first mention order
for kind in ("Figure", "Table"):
    seen = []
    for m in re.finditer(rf"(?<!\*\*){kind} (\d+)", re.sub(r"!\[.*?\]\(.*?\)|^Table:.*$", "", body, flags=re.M)):
        n = int(m.group(1))
        if n not in seen: seen.append(n)
    print(kind, "first-mention order:", seen)
    caps = re.findall(rf"\*\*{kind} (\d+)\.\*\*", body); print(kind, "captions:", caps)
# citations vs references
ref_entries = [r.strip() for r in refs.strip().split("\n\n") if r.strip()]
ref_keys = set()
for r in ref_entries:
    m = re.match(r"^([^,(]+?)(?:,|\s)[^()]*?\((\d{4})\)", r)
    ref_keys.add((m.group(1).strip(), m.group(2)))
txt = re.sub(r"\n", " ", body)
cites = set()
for m in re.finditer(r"([A-Z][A-Za-zäöüéèíóúšćžđ'\-]+(?: and [A-Z][A-Za-z\-]+)?(?: et al\.)?),? \(?(\d{4})(?:, preprint)?\)?", txt):
    cites.add((m.group(1), m.group(2)))
# also handle 'Livesey and Marsh (2023, 2025)' and '(Notin et al., 2023, preprint)'
extra = [("Livesey", "2023"), ("Livesey", "2025")]
def first(a):
    a = a.replace(" et al.", ""); a = a.split(" and ")[0]; return a
cited = {(first(a), y) for a, y in cites}
cited |= set(extra)
refk = {(a, y) for a, y in ref_keys}
refk_first = {(a, y) for a, y in refk}
# match by surname prefix (references hold the first author surname)
missing = [(a, y) for a, y in sorted(cited) if not any(a == r or a.startswith(r) or r.startswith(a) for r, yy in refk if yy == y)]
print("cited but not in references:", missing)
unused = [(r, y) for r, y in sorted(refk) if not any(r.startswith(a) or a.startswith(r) for a, yy in cited if yy == y)]
print("references never cited:", unused)
print("n references:", len(ref_entries))
