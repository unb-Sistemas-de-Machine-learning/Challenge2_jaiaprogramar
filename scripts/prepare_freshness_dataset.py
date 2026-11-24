#!/usr/bin/env python3
"""Validate and normalize AgriFreshNET or another freshness image dataset.

Expected input layout (label directory may be nested under product):
    INPUT/<product>/<fresh-or-semi-fresh-or-rotten>/<image>
    INPUT/<fresh-or-semi-fresh-or-rotten>_<product>/<image>
    INPUT/<fresh-or-semi-fresh-or-rotten>/<image>

Files are copied, never moved or deleted from the source. Corrupt and duplicate
images are omitted from the output and recorded in the report/manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABEL_ALIASES = {
    "produto_fresco": "produto_fresco", "fresh": "produto_fresco", "ripe": "produto_fresco",
    "fresco": "produto_fresco", "fresh_fruits": "produto_fresco", "fresh_vegetables": "produto_fresco",
    "produto_estragado": "produto_estragado", "spoiled": "produto_estragado", "rotten": "produto_estragado",
    "estragado": "produto_estragado", "rotten_fruits": "produto_estragado", "rotten_vegetables": "produto_estragado",
}

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print("Dependência ausente: instale Pillow com `python -m pip install Pillow`.", file=sys.stderr)
    raise SystemExit(2)


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return value or "produto_nao_informado"


def image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            yield path


def identify(path: Path, root: Path) -> tuple[str, str] | None:
    parts = path.relative_to(root).parts[:-1]
    for index in range(len(parts) - 1, -1, -1):
        component = parts[index].lower().replace("-", "_").replace(" ", "_")
        candidate = LABEL_ALIASES.get(component)
        if candidate:
            product = parts[index - 1] if index > 0 else "produto_nao_informado"
            return candidate, slug(product)

        tokens = [token for token in component.split("_") if token]
        stage_tokens = {"fresh", "fresco", "ripe", "semi", "semifresh", "rotten", "spoiled", "estragado"}
        has_semi = "semi" in tokens or "semifresh" in tokens
        has_fresh = "fresh" in tokens or "fresco" in tokens or "ripe" in tokens
        has_spoiled = "rotten" in tokens or "spoiled" in tokens or "estragado" in tokens
        if has_fresh or has_semi or has_spoiled:
            label = "produto_fresco" if has_fresh and not has_semi and not has_spoiled else "produto_estragado"
            product_tokens = [token for token in tokens if token not in stage_tokens]
            product = "_".join(product_tokens) if product_tokens else (
                parts[index - 1] if index > 0 else "produto_nao_informado"
            )
            return label, slug(product)
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Diretório original, organizado por produto/rótulo ou rótulo")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório organizado de saída")
    parser.add_argument("--source", required=True, help="Identificador curto da fonte, ex.: kaggle_fvd_2026 ou fotos_equipe")
    parser.add_argument("--manifest", type=Path, help="CSV de saída; padrão: <output-dir>/manifest.csv")
    args = parser.parse_args()

    source_root = args.input_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else output_root / "manifest.csv"
    if not source_root.is_dir():
        parser.error(f"Diretório de entrada não existe: {source_root}")
    if source_root == output_root or source_root in output_root.parents:
        parser.error("A saída não pode ser o próprio diretório de entrada nem estar dentro dele.")

    output_root.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    counts = {"copied": 0, "corrupt": 0, "duplicate": 0, "unlabeled": 0, "unsupported": 0}
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            counts["unsupported"] += 1
            continue
        classification = identify(path, source_root)
        if classification is None:
            counts["unlabeled"] += 1
            continue
        label, product = classification
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            counts["corrupt"] += 1
            rows.append({"image_path": "", "label": label, "source": args.source, "product": product,
                         "status": "corrupt", "sha256": "", "notes": str(exc)[:240]})
            continue
        digest = sha256(path)
        if digest in seen:
            counts["duplicate"] += 1
            rows.append({"image_path": "", "label": label, "source": args.source, "product": product,
                         "status": "duplicate", "sha256": digest, "notes": "SHA-256 idêntico a imagem já processada"})
            continue
        seen.add(digest)
        destination_dir = output_root / label / product
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{digest[:16]}{path.suffix.lower()}"
        shutil.copy2(path, destination)
        rows.append({"image_path": destination.relative_to(output_root).as_posix(), "label": label,
                     "source": args.source, "product": product, "status": "accepted", "sha256": digest, "notes": ""})
        counts["copied"] += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image_path", "label", "source", "product", "status", "sha256", "notes"]
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Concluído: {counts['copied']} imagens aceitas; {counts['corrupt']} corrompidas; "
          f"{counts['duplicate']} duplicadas; {counts['unlabeled']} sem rótulo reconhecido; "
          f"{counts['unsupported']} extensões ignoradas.")
    print(f"Manifesto: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
