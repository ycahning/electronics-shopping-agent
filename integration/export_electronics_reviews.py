"""Build a reproducible review index from the official gzip prefix or local full archive."""

import argparse
import csv
import gzip
import hashlib
import json
import zlib
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

SOURCE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz"
PREFIX_BYTES = 32 * 1024 * 1024
MAX_EXCERPTS_PER_PRODUCT = 4
START_MS = 1577836800000  # 2020-01-01 UTC
END_MS = 1704067200000  # 2024-01-01 UTC, exclusive


def complete_rows(prefix: bytes) -> Iterator[tuple[int, dict]]:
    decoder = zlib.decompressobj(wbits=31)
    remainder = b""
    line_number = 0
    for offset in range(0, len(prefix), 64 * 1024):
        lines = (
            remainder + decoder.decompress(prefix[offset : offset + 64 * 1024])
        ).split(b"\n")
        remainder = lines.pop()
        for line in lines:
            line_number += 1
            yield line_number, json.loads(line)


def export(
    prefix_path: Path, catalog_path: Path, *, prefix_bytes: int = PREFIX_BYTES
) -> dict:
    if prefix_bytes <= 0:
        raise ValueError("The requested prefix size must be positive.")
    with prefix_path.open("rb") as stream:
        prefix = stream.read(prefix_bytes + 1)
    if len(prefix) != prefix_bytes:
        raise ValueError(
            f"Expected exactly {prefix_bytes} compressed bytes from the start of the source."
        )
    return _index(
        complete_rows(prefix),
        catalog_path,
        {
            "prefix_bytes": len(prefix),
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        },
    )


def export_archive(
    archive_path: Path, catalog_path: Path, *, excerpts_per_product: int = 8
) -> dict:
    """Stream a local full gzip and retain a bounded collection of passages per ASIN."""
    if not 1 <= excerpts_per_product <= 12:
        raise ValueError("The archive export can retain 1 to 12 passages per product.")
    with archive_path.open("rb") as archive:
        digest = hashlib.file_digest(archive, "sha256").hexdigest()
        archive.seek(0)
        with gzip.GzipFile(fileobj=archive) as reviews:
            return _index(
                ((line, json.loads(row)) for line, row in enumerate(reviews, 1)),
                catalog_path,
                {
                    "archive_sha256": digest,
                    "excerpts_per_product": excerpts_per_product,
                },
                excerpts_per_product=excerpts_per_product,
                word_limit=72,
                character_limit=600,
            )


def _index(
    corpus: Iterator[tuple[int, dict]],
    catalog_path: Path,
    provenance: dict,
    *,
    excerpts_per_product: int = MAX_EXCERPTS_PER_PRODUCT,
    word_limit: int = 24,
    character_limit: int = 240,
) -> dict:
    with catalog_path.open("rb") as stream:
        catalog_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    with catalog_path.open(encoding="utf-8", newline="") as stream:
        products = {
            item["parent_asin"]
            for item in csv.DictReader(stream)
            if item["title"].strip()
        }
    rows = 0
    grouped: dict[str, list] = {}
    for line_number, row in corpus:
        rows += 1
        asin = row["parent_asin"]
        if (
            asin not in products
            or not START_MS <= row["timestamp"] < END_MS
            or not isinstance(row.get("text"), str)
        ):
            continue
        entries = grouped.get(asin)
        if entries is not None and len(entries) >= excerpts_per_product:
            continue
        # Preserve source words. Account IDs and other fields are discarded.
        words = row["text"].split()
        text = " ".join(words[:word_limit])[:character_limit].strip()
        if len(text) < 12:
            continue
        entries = grouped.setdefault(asin, [])
        if any(entry[1] == text for entry in entries):
            continue
        entries.append([line_number, text])
    if not grouped:
        raise ValueError("The sampled records did not match the catalog.")
    return {
        "schema_version": 1,
        "source": SOURCE,
        **provenance,
        "catalog_sha256": catalog_sha256,
        "complete_source_rows": rows,
        "timestamp_interval_ms": [START_MS, END_MS],
        "excerpts": grouped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefix-bytes", type=int, default=PREFIX_BYTES)
    parser.add_argument(
        "--full-archive",
        action="store_true",
        help="Stream a complete local Electronics gzip; no download is performed.",
    )
    args = parser.parse_args()
    if args.output.resolve() in {args.source.resolve(), args.catalog.resolve()}:
        parser.error("Output must differ from the source archive and catalog paths.")
    data = (
        export_archive(args.source, args.catalog)
        if args.full_archive
        else export(args.source, args.catalog, prefix_bytes=args.prefix_bytes)
    )
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=args.output.parent, suffix=".review.tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            if args.full_archive:
                with gzip.GzipFile(
                    filename="", fileobj=stream, mode="wb", mtime=0
                ) as compressed:
                    encoder = json.JSONEncoder(
                        ensure_ascii=False, separators=(",", ":")
                    )
                    for chunk in encoder.iterencode(data):
                        compressed.write(chunk.encode())
            else:
                payload = json.dumps(
                    data, ensure_ascii=False, separators=(",", ":")
                ).encode()
                stream.write(gzip.compress(payload, mtime=0))
        temporary.replace(args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
