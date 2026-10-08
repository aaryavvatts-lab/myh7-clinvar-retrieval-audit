"""v7 step: per-gene AlphaFold DB model (by UniProt) and gnomAD v4 variants (GraphQL, GRCh38).
Raw responses are cached (data/external/alphafold, data/raw/gnomad); re-runs reuse the cache
unless FORCE_DOWNLOAD=1. Provenance (versions, URLs, sha256, UTC time) -> logs/provenance_acquire_v7.json
"""
from __future__ import annotations
import datetime as dt, json, os, time
import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBParser
from src.genes import *   # noqa
from src.mg_clinvar import sha256

PROV = LOGS / "provenance_acquire_v7.json"
FORCE = bool(os.environ.get("FORCE_DOWNLOAD"))
AA3to1 = AA3
THREE = {v: k for k, v in AA3.items()}


def _post(url, **kw):
    for a in range(5):
        try:
            r = requests.post(url, timeout=kw.pop("timeout", 600), **kw)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa
            print("   retry", a, str(e)[:100]); time.sleep(5 * (a + 1))
    raise RuntimeError(f"POST failed {url}")


def af_model(gene):
    """Download the AlphaFold DB F1 model for the gene's UniProt accession; returns provenance dict."""
    up = UNIPROT[gene]
    path = EXT / "alphafold" / f"AF-{up}-F1-model.pdb"
    info = {"gene": gene, "uniprot": up}
    r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{up}", timeout=60)
    if r.status_code != 200 or not r.json():
        info.update(available=False, reason=f"AlphaFold DB API returned HTTP {r.status_code} / empty for {up}")
        return info
    ents = r.json()
    f1 = [e for e in ents if e["entryId"] == f"AF-{up}-F1"]
    if not f1:
        info.update(available=False, reason="no canonical F1 entry"); return info
    e = f1[0]
    info.update(available=True, entryId=e["entryId"], model_version=e.get("latestVersion"),
                uniprotStart=e["uniprotStart"], uniprotEnd=e["uniprotEnd"], seq_len=len(e["uniprotSequence"]),
                pdbUrl=e["pdbUrl"], model_created=e.get("modelCreatedDate"), sequence=e["uniprotSequence"])
    info["full_length_single_fragment"] = bool(e["uniprotStart"] == 1 and e["uniprotEnd"] == len(e["uniprotSequence"]))
    if FORCE or not path.exists():
        with requests.get(e["pdbUrl"], timeout=120) as rr:
            rr.raise_for_status(); path.write_bytes(rr.content)
    info["pdb_path"] = str(path); info["pdb_sha256"] = sha256(path)
    return info


def gnomad(gene):
    path = RAW / "gnomad" / f"{gene}_gnomad_r4.parquet"
    if path.exists() and not FORCE:
        return pd.read_parquet(path), {"cached": True}
    pops = "populations{id ac an}" if gene == "MYH7" else ""
    faf = "faf95{popmax popmax_population}" if gene == "MYH7" else ""
    q = ('query{ gene(gene_symbol:"%s", reference_genome:GRCh38){ variants(dataset:gnomad_r4){ variant_id pos ref alt '
         'consequence hgvsp transcript_id exome{ac an af %s %s} genome{ac an af %s %s} } } }' % (gene, pops, faf, pops, faf))
    r = _post("https://gnomad.broadinstitute.org/api", json={"query": q})
    j = r.json()
    assert "errors" not in j, j.get("errors")
    rows = []
    for v in j["data"]["gene"]["variants"]:
        row = {"variant_id38": v["variant_id"], "pos38": v["pos"], "ref": v["ref"], "alt": v["alt"],
               "consequence": v["consequence"], "hgvsp": v["hgvsp"], "transcript": v["transcript_id"]}
        for ds in ("exome", "genome"):
            d = v.get(ds) or {}
            row[f"ac_{ds}"] = d.get("ac"); row[f"an_{ds}"] = d.get("an"); row[f"af_{ds}"] = d.get("af")
            if gene == "MYH7":
                row[f"faf95_{ds}"] = (d.get("faf95") or {}).get("popmax")
                for p in d.get("populations") or []:
                    row[f"{ds}_{p['id']}_ac"] = p["ac"]; row[f"{ds}_{p['id']}_an"] = p["an"]
        rows.append(row)
    g = pd.DataFrame(rows)
    g.to_parquet(path, index=False)
    return g, {"cached": False, "retrieved_utc": dt.datetime.utcnow().isoformat() + "Z", "n_variants": int(len(g))}


def main(genes=None):
    genes = genes or (GENES + EXTRA_GENES)
    prov = json.loads(PROV.read_text()) if PROV.exists() else {}
    for g in genes:
        t = time.time()
        info = af_model(g)
        gm, gi = gnomad(g)
        info = {k: v for k, v in info.items() if k != "sequence"} | {"gnomad": gi}
        mm = gm[gm.consequence == "missense_variant"]
        info["gnomad_n_variants"] = int(len(gm)); info["gnomad_n_missense"] = int(len(mm))
        prov[g] = info
        PROV.write_text(json.dumps(prov, indent=1, default=str))
        print(f"[acquire] {g}: AF available={info['available']} len={info.get('seq_len')} "
              f"gnomAD variants={len(gm)} missense={len(mm)} ({time.time()-t:.0f}s)", flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
