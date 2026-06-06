"""Build a diverse PDF corpus from arXiv, PMC and CORE.

Secrets and contact details are read from the environment (``.env`` via
pydantic-settings) — never hardcoded. Copy ``.env.example`` to ``.env`` and fill
in your CORE API key and Entrez email before running the PMC/CORE sources.

This is a courteous scraper: it honours each provider's rate limits. Run it only
for content you are permitted to download.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import requests
from pydantic_settings import BaseSettings, SettingsConfigDict

from flashmask.config import paths

ARXIV_CATEGORIES = ["cs.LG", "cs.CV", "stat.ML", "cs.AI"]
PMC_QUERIES = ["cancer", "genomics", "immunology", "neuroscience", "public health"]
CORE_QUERIES = ["machine learning", "climate change", "economics", "psychology", "chemistry"]


class ScrapeSettings(BaseSettings):
    """Credentials loaded from the environment / ``.env`` (prefix ``FLASHMASK_``)."""

    model_config = SettingsConfigDict(env_prefix="FLASHMASK_", env_file=".env", extra="ignore")

    core_api_key: str = ""
    entrez_email: str = ""


def fetch_arxiv(category: str, n: int, out_dir: Path, *, batch: int = 100) -> int:
    """Download up to ``n`` PDFs from an arXiv category (1 request / 3 s)."""
    import feedparser

    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded, start = 0, 0
    while downloaded < n:
        want = min(batch, n - downloaded)
        feed = feedparser.parse(
            f"http://export.arxiv.org/api/query?search_query=cat:{category}"
            f"&start={start}&max_results={want}"
        )
        if not feed.entries:
            break
        for entry in feed.entries:
            if downloaded >= n:
                break
            arxiv_id = entry.id.split("/abs/")[-1]
            dest = out_dir / f"{arxiv_id.replace('/', '_')}.pdf"
            if not dest.exists():
                pdf_url = entry.link.replace("abs", "pdf") + ".pdf"
                resp = requests.get(pdf_url, timeout=60)
                if resp.ok:
                    dest.write_bytes(resp.content)
                time.sleep(3)  # arXiv asks for >= 3 s between requests
            downloaded += 1
        start += want
    return downloaded


def fetch_pmc(query: str, n: int, out_dir: Path, email: str, *, batch: int = 200) -> int:
    """Download up to ``n`` open-access PDFs from PubMed Central."""
    from Bio import Entrez

    if not email:
        raise ValueError("Set FLASHMASK_ENTREZ_EMAIL in your .env (NCBI requires a contact email).")
    Entrez.email = email
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded, retstart = 0, 0
    while downloaded < n:
        retmax = min(batch, n - downloaded)
        with Entrez.esearch(db="pmc", term=query, retmax=retmax, retstart=retstart) as handle:
            ids = Entrez.read(handle).get("IdList", [])
        if not ids:
            break
        for pmcid in ids:
            if downloaded >= n:
                break
            dest = out_dir / f"PMC{pmcid}.pdf"
            if not dest.exists():
                url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
                resp = requests.get(url, timeout=60)
                if resp.ok and "pdf" in resp.headers.get("Content-Type", ""):
                    dest.write_bytes(resp.content)
                time.sleep(0.34)  # ~3 requests/s
            downloaded += 1
        retstart += retmax
    return downloaded


def fetch_core(query: str, n: int, out_dir: Path, api_key: str, *, page_size: int = 100) -> int:
    """Download up to ``n`` PDFs from CORE (requires an API key)."""
    if not api_key:
        raise ValueError("Set FLASHMASK_CORE_API_KEY in your .env to use the CORE source.")
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {api_key}"}
    downloaded = 0
    for page in range(1, math.ceil(n / page_size) + 1):
        if downloaded >= n:
            break
        resp = requests.get(
            "https://api.core.ac.uk/v3/search/works",
            params={"q": query, "page": page, "pageSize": page_size},
            headers=headers,
            timeout=60,
        )
        for work in resp.json().get("results", []):
            if downloaded >= n:
                break
            pdf_url, doi = work.get("downloadUrl"), (work.get("doi") or "").replace("/", "_")
            if pdf_url and doi:
                dest = out_dir / f"{doi}.pdf"
                if not dest.exists():
                    r = requests.get(pdf_url, timeout=60)
                    if r.ok:
                        dest.write_bytes(r.content)
                    time.sleep(1)
                downloaded += 1
        time.sleep(1)
    return downloaded


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Scrape a PDF corpus from arXiv / PMC / CORE.")
    ap.add_argument("--sources", nargs="+", default=["arxiv"], choices=["arxiv", "pmc", "core"])
    ap.add_argument("--per-query", type=int, default=50, help="PDFs to fetch per category/query")
    ap.add_argument("--out", type=Path, default=paths.data_raw)
    args = ap.parse_args(argv)

    cfg = ScrapeSettings()
    if "arxiv" in args.sources:
        for cat in ARXIV_CATEGORIES:
            print(f"arXiv:{cat} -> {fetch_arxiv(cat, args.per_query, args.out / 'arxiv')}")
    if "pmc" in args.sources:
        for q in PMC_QUERIES:
            print(f"PMC:{q} -> {fetch_pmc(q, args.per_query, args.out / 'pmc', cfg.entrez_email)}")
    if "core" in args.sources:
        for q in CORE_QUERIES:
            print(
                f"CORE:{q} -> {fetch_core(q, args.per_query, args.out / 'core', cfg.core_api_key)}"
            )


if __name__ == "__main__":
    main()
