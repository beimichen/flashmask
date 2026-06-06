"""Render PDF pages to PNG images (PyMuPDF).

Importable :func:`pdf_to_images` plus a small ``python -m`` CLI. Replaces the old
``pdf_to_images.py`` / the conversion half of ``download_arxiv_pdfs.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    """Render every page of ``pdf_path`` to ``output_dir/<stem>/<stem>_page_N.png``."""
    import fitz  # PyMuPDF — part of the optional [scrape] extra

    pdf_path, output_dir = Path(pdf_path), Path(output_dir)
    page_dir = output_dir / pdf_path.stem
    page_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            out = page_dir / f"{pdf_path.stem}_page_{i}.png"
            page.get_pixmap(dpi=dpi).save(out)
            written.append(out)
    return written


def convert_folder(input_folder: Path, output_folder: Path, dpi: int = 200) -> int:
    """Convert every PDF under ``input_folder``. Returns the number of pages written."""
    input_folder, output_folder = Path(input_folder), Path(output_folder)
    pages = 0
    for pdf in sorted(input_folder.rglob("*.pdf")):
        try:
            pages += len(pdf_to_images(pdf, output_folder, dpi))
        except Exception as exc:  # noqa: BLE001 - keep going on a single bad PDF
            print(f"  ! failed {pdf}: {exc}")
    return pages


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Convert all PDFs in a folder to page images.")
    ap.add_argument("input_folder", type=Path)
    ap.add_argument("output_folder", type=Path)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)
    n = convert_folder(args.input_folder, args.output_folder, args.dpi)
    print(f"Wrote {n} page images to {args.output_folder}")


if __name__ == "__main__":
    main()
