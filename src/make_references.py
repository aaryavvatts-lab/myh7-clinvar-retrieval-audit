"""Build the reference list for the v6 paper from Crossref metadata (every DOI resolved, not typed from memory).

Reads/writes:  paper/references_crossref.json (cache),  paper/references.md (APA 7 style, volume-year convention)
Run from v6/:  ../.venv/bin/python -m src.make_references [--refresh]
Year convention: year of the volume/print issue; online-first year only when no print date exists.
Preprints are marked [Preprint]. Entries are listed alphabetically by first author.
"""
from __future__ import annotations
import html, json, re, sys, time, urllib.request
from pathlib import Path

DOIS = """10.1038/nmeth0410-248 10.1038/s41594-022-00849-w 10.1073/pnas.2304302120 10.1016/j.gim.2025.101402
10.1038/s41588-023-01465-0 10.1126/science.adg7492 10.1016/j.jacc.2021.07.017 10.1016/j.ajhg.2023.11.005
10.1161/CIRCGEN.123.004377 10.1002/humu.22768 10.1016/j.ajhg.2016.08.016 10.1073/pnas.2002660117
10.1016/j.jmb.2019.04.009 10.1101/2024.05.29.596373 10.1038/s41586-020-2308-7 10.1038/gim.2017.218
10.1093/nar/gkx1153 10.1186/s12864-025-11534-9 10.1016/j.ajhg.2023.08.003 10.1161/CIRCULATIONAHA.124.069398
10.1093/nar/gkg509 10.1016/j.ajhg.2022.10.013 10.1101/gr.275528.121 10.1371/journal.pcbi.1009818
10.1016/j.ajhg.2022.01.006 10.1007/s00439-025-02732-2 10.1093/nar/gky1016 10.1038/gim.2015.30
10.1038/gim.2017.210 10.1016/j.ajhg.2024.07.018 10.1161/CIRCULATIONAHA.119.042339 10.1161/CIRCULATIONAHA.123.065987
10.1101/gr.226589.117 10.1038/s41586-023-06045-0 10.1186/s13059-025-03575-w 10.15252/msb.202211474
10.1101/2023.12.07.570727 10.1101/2025.10.07.680947 10.1186/s13073-024-01358-9 10.1101/2024.04.11.588920
10.1161/JAHA.124.036921 10.1038/s44318-024-00199-x 10.1186/s13073-024-01392-7 10.1073/pnas.1606950113
10.1161/JAHA.125.041298 10.1186/s13073-020-00803-9 10.1186/s13059-016-0953-9 10.1038/s41586-021-03819-2
10.1093/nar/gkab1061 10.1016/j.ajhg.2018.08.005 10.1186/1471-2164-14-S3-S3 10.1093/hmg/ddu733
10.1101/gr.097857.109 10.1148/radiology.143.1.7063747 10.1038/s41592-019-0686-2 10.1038/s41586-020-2649-2
10.1093/bioinformatics/btp163 10.1093/nar/gkac1052 10.1002/humu.24088 10.1016/j.devcel.2009.10.013""".split()
PAGES = {"10.1161/CIRCGEN.123.004377": "e004377"}     # Crossref omits the article number
PREPRINT = {"10.1101/2024.05.29.596373", "10.1101/2023.12.07.570727", "10.1101/2025.10.07.680947", "10.1101/2024.04.11.588920"}
CACHE = Path("paper/references_crossref.json")


def fetch(d):
    req = urllib.request.Request("https://api.crossref.org/works/" + d, headers={"User-Agent": "myh7-paper-build/1.0 (reference check)"})
    return json.load(urllib.request.urlopen(req, timeout=30))["message"]


def initials(given):
    parts = re.split(r"[\s\-‐]+", given.replace(".", " ").strip())
    out = []
    for p in parts:
        if p:
            out.append(p[0] + ".")
    return " ".join(out) if "-" not in given else "-".join(x[0] + "." for x in re.split(r"[\-‐]", given.replace(".", " ").strip()) if x)


def name(a):
    fam = a.get("family", "").strip(); giv = a.get("given", "").strip()
    if fam == "Pardo" and giv.startswith("Eduard Porta"):
        fam, giv = "Porta-Pardo", "Eduard"
    return f"{fam}, {initials(giv)}" if giv and giv != "?" else fam


AUTH_BY_DOI = {
    "10.1093/nar/gkg509": "Ng, P. C., & Henikoff, S.",
    "10.1016/j.ajhg.2022.10.013": "Pejaver, V., Byrne, A. B., Feng, B.-J., Pagel, K. A., Mooney, S. D., Karchin, R., O'Donnell-Luria, A., Harrison, S. M., Tavtigian, S. V., Greenblatt, M. S., Biesecker, L. G., Radivojac, P., Brenner, S. E., & ClinGen Sequence Variant Interpretation Working Group",
    "10.1016/j.gim.2025.101402": "Bergquist, T., Stenton, S. L., Nadeau, E. A. W., Byrne, A. B., Greenblatt, M. S., Harrison, S. M., Tavtigian, S. V., O'Donnell-Luria, A., Biesecker, L. G., Radivojac, P., Brenner, S. E., Pejaver, V., & ClinGen Sequence Variant Interpretation Working Group",
    "10.1186/s13073-024-01358-9": "Zhang, X., Theotokis, P. I., Li, N., SHaRe Investigators, Wright, C. F., Samocha, K. E., Whiffin, N., & Ware, J. S.",
    "10.1093/nar/gkac1052": "UniProt Consortium.",
}
YEAR_BY_DOI = {"10.1101/2024.04.11.588920": 2026}
AUTH_OVERRIDE = {"UniProt: the Universal Protein Knowledgebase in 2023": "UniProt Consortium"}


def authors(msg):
    if clean(msg.get("title", [""])[0]) in AUTH_OVERRIDE:
        return AUTH_OVERRIDE[clean(msg["title"][0])]
    al = [a for a in msg.get("author", []) if a.get("family") and a.get("family") != "?"]
    if not al:
        return "UniProt Consortium"
    names = [name(a) for a in al]
    if len(names) == 1:
        return names[0]
    if len(names) <= 20:
        return ", ".join(names[:-1]) + ", & " + names[-1]
    return ", ".join(names[:19]) + ", ... " + names[-1]


def year(msg):
    for k in ("published-print", "issued"):
        dp = msg.get(k, {}).get("date-parts", [[None]])[0]
        if dp and dp[0]:
            return dp[0]


PROTECT = {"College", "AlphaFold", "AlphaFold2", "AlphaMissense", "Mendelian", "MYH7", "ACMG/AMP", "ClinVar", "ClinPred", "ClinGen", "DNA", "SNVs",
           "ROC", "American", "Medical", "Genetics", "Genomics", "Association", "Molecular", "Pathology", "DNA", "Python", "SciPy", "NumPy",
           "Biopython", "I", "II", "PP3/BP4", "PS3/BS3", "AMP", "ACMG", "VUS", "REVEL", "CADD", "SIFT", "PolyPhen-2", "ProteinGym", "Scikit-learn",
           "Nonsynonymous", "Alzheimer", "PLOS", "SNV", "MAVE", "HCM", "Circulation:"}


def sentence_case(t):
    words = t.split(" ")
    big = [w for w in words if len(w) > 3 and w[0].isupper()]
    long_ = [w for w in words if len(w) > 3]
    if not long_ or len(big) / len(long_) < 0.5:
        return t                       # already sentence case
    out = []; after_colon = True
    for i, w in enumerate(words):
        core = w.strip("():,")
        keep = (i == 0) or after_colon or core in PROTECT or bool(re.search(r"[A-Z].*[A-Z]|\d", core)) or core.startswith("*")
        if "-" in w and re.search(r"[A-Z].*-[A-Z]", core) and core not in PROTECT and i > 0 and not after_colon:
            keep = False
        w2 = w if keep else w[0].lower() + w[1:]
        if "-" in w2 and not keep:
            h = w2.split("-"); w2 = h[0] + "-" + "-".join(x[0].lower() + x[1:] if x and x not in PROTECT else x for x in h[1:])
        elif "-" in w and keep and i > 0 and not after_colon and core not in PROTECT:
            pass
        out.append(w2)
        after_colon = w.endswith(":")
    return " ".join(out)


def clean(t):
    t = html.unescape(re.sub(r"<[^>]+>", "", t)); return re.sub(r"\s+", " ", t).strip()


def ascii_punct(t):
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u2010", "-"), ("\u2011", "-"), ("\u201c", '"'), ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "-")):
        t = t.replace(a, b)
    return t


def entry(d, msg):
    msg = dict(msg); msg["_doi"] = d
    title = sentence_case(clean(msg["title"][0])).rstrip(".")
    if d == "10.1161/CIRCGEN.123.004377":
        title = title.replace("MYH7", "*MYH7*")
    cont = clean((msg.get("container-title") or [""])[0])
    vol = msg.get("volume"); pg = PAGES.get(d) or msg.get("page") or msg.get("article-number")
    if pg and re.match(r"^MSB", str(pg)): pg = str(pg)[3:].lower().replace("202211474", "e11474") if False else "e11474"
    y = year(msg)
    a = AUTH_BY_DOI.get(d) or authors(msg)
    if a.endswith("Group"):
        a += "."
    y = YEAR_BY_DOI.get(d, y)
    if d in PREPRINT:
        ver = " (version 3)" if d == "10.1101/2024.04.11.588920" else ""
        return ascii_punct(f"{a} ({y}). {title}. *bioRxiv*{ver}. [Preprint] https://doi.org/{d}")
    src = f"*{cont}*" + (f", *{vol}*" if vol else "") + (f", {pg}" if pg else "")
    for a_, b_ in (("Large-Scale", "Large-scale"), ("Meta-Analysis", "Meta-analysis"), ("Nonsynonymous single", "nonsynonymous single"),
                   ("UniProt: the universal", "UniProt: The universal")):
        title = title.replace(a_, b_)
    ttl = title if title[-1] in "?!" else title + "."
    return ascii_punct(f"{a} ({y}). {ttl} {src}. https://doi.org/{d}")


def main():
    refresh = "--refresh" in sys.argv
    cache = json.loads(CACHE.read_text()) if CACHE.exists() and not refresh else {}
    for d in DOIS:
        if d not in cache:
            cache[d] = fetch(d); time.sleep(0.2)
    CACHE.write_text(json.dumps(cache))
    items = []
    for d in DOIS:
        msg = cache[d]
        import unicodedata
        key = unicodedata.normalize("NFKD", (AUTH_BY_DOI.get(d) or authors(msg)).lower()).encode("ascii", "ignore").decode()
        items.append((key, YEAR_BY_DOI.get(d, year(msg)), entry(d, msg)))
    items.append(("pedregosa, f.", 2011, "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html"))
    items.sort(key=lambda x: (x[0], x[1]))
    # UniProt consortium sorts under U
    Path("paper/references.md").write_text("\n\n".join(e for _, _, e in items) + "\n")
    print(len(items), "references written")


if __name__ == "__main__":
    main()
