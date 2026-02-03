#!/usr/bin/env python3
"""Extract ethnicity-language and ethnicity-religion tables from NBS xlsx.

This script reads a dense multi-sheet XLSX and outputs normalized
JSON tables suitable for moldova_personas generation.

Usage:
  python tools/scripts/extract_ethnocultural_xlsx.py \
    --input /path/to/Date_Comunicat_Etnoculturale_20_10_25.xlsx \
    --output config/ethnocultural_tables_2024.json
"""

from __future__ import annotations

import argparse
import json
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import unicodedata


NS_MAIN = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _normalize(text: str) -> str:
    text = " ".join((text or "").strip().lower().split())
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s*/\s*", "/", text)
    return text


def _load_shared_strings(z: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    strings: List[str] = []
    for si in root.findall("ns:si", NS_MAIN):
        parts = [t.text or "" for t in si.findall(".//ns:t", NS_MAIN)]
        strings.append("".join(parts))
    return strings


def _col_to_index(col: str) -> int:
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _split_ref(cell_ref: str) -> Tuple[str, int]:
    col = "".join([c for c in cell_ref if c.isalpha()])
    row = "".join([c for c in cell_ref if c.isdigit()])
    return col, int(row)


def _load_sheet_cells(
    z: zipfile.ZipFile, sheet_path: str, shared_strings: List[str]
) -> Dict[int, Dict[int, str]]:
    data: Dict[int, Dict[int, str]] = defaultdict(dict)
    root = ET.fromstring(z.read(sheet_path))
    for cell in root.findall(".//ns:c", NS_MAIN):
        ref = cell.attrib.get("r")
        if not ref:
            continue
        col, row = _split_ref(ref)
        col_idx = _col_to_index(col)
        v = cell.find("ns:v", NS_MAIN)
        if v is None:
            continue
        value = v.text or ""
        if cell.attrib.get("t") == "s":
            try:
                value = shared_strings[int(value)]
            except Exception:
                pass
        data[row][col_idx] = value
    return data


def _find_sheet_path(z: zipfile.ZipFile, sheet_name: str) -> Optional[str]:
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall(f"{{{NS_REL}}}Relationship")}
    for sheet in wb.findall("ns:sheets/ns:sheet", NS_MAIN):
        name = sheet.attrib.get("name")
        if name != sheet_name:
            continue
        r_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rel_map.get(r_id)
        if not target:
            return None
        if not target.startswith("xl/"):
            target = "xl/" + target
        return target
    return None


def _extract_table(
    cells: Dict[int, Dict[int, str]],
    header_labels: List[str],
    row_label_aliases: List[str],
    stop_after_blank: int = 3,
) -> Tuple[List[str], List[Tuple[str, List[float]]]]:
    # Find header row and label column
    header_row = None
    label_col = None
    target_labels = set(_normalize(x) for x in row_label_aliases)
    for row_idx in sorted(cells.keys()):
        row = cells[row_idx]
        for col_idx, val in row.items():
            if _normalize(val) in target_labels:
                header_row = row_idx
                label_col = col_idx
                break
        if header_row is not None:
            break
    if header_row is None or label_col is None:
        raise ValueError("Could not locate header row with label column")

    # Determine column headers from header_row, starting to the right of label_col
    headers: List[str] = []
    header_cols: List[int] = []
    row = cells.get(header_row, {})
    for col_idx in sorted(row.keys()):
        if col_idx <= label_col:
            continue
        val = str(row[col_idx]).strip()
        if not val:
            continue
        headers.append(val)
        header_cols.append(col_idx)

    if not headers:
        raise ValueError("No headers found to the right of label column")

    # Parse data rows
    rows_out: List[Tuple[str, List[float]]] = []
    blanks = 0
    for row_idx in range(header_row + 1, max(cells.keys()) + 1):
        row = cells.get(row_idx, {})
        label = str(row.get(label_col, "")).strip()
        if not label:
            blanks += 1
            if blanks >= stop_after_blank:
                break
            continue
        blanks = 0

        # Skip non-data lines
        norm_label = _normalize(label)
        if norm_label in {"total", "populația totală", "populatia totală", "populatia total", "populația total"}:
            continue
        if norm_label.startswith("figura"):
            break

        values: List[float] = []
        has_any = False
        for col_idx in header_cols:
            raw = str(row.get(col_idx, "")).strip()
            if raw == "":
                values.append(0.0)
                continue
            try:
                values.append(float(raw))
                has_any = True
            except ValueError:
                values.append(0.0)
        if has_any:
            rows_out.append((label, values))

    return headers, rows_out


def _normalize_dist(values: Dict[str, float]) -> Dict[str, float]:
    total = sum(values.values())
    if total <= 0:
        return {k: 0.0 for k in values}
    return {k: v / total for k, v in values.items()}


def _build_ethnicity_table(
    headers: List[str],
    rows: List[Tuple[str, List[float]]],
    col_map: Dict[str, str],
    row_map: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}

    for row_label, vals in rows:
        row_key = row_map.get(row_label, None)
        if not row_key:
            # try normalized match
            row_key = row_map.get(_normalize(row_label), None)
        if not row_key:
            # Skip rows we don't map (e.g., undeclared)
            continue

        agg: Dict[str, float] = defaultdict(float)
        for header, val in zip(headers, vals):
            col_key = col_map.get(header, None)
            if not col_key:
                col_key = col_map.get(_normalize(header), None)
            if not col_key:
                continue
            # values are percentages (0-100)
            agg[col_key] += float(val) / 100.0

        table[row_key] = _normalize_dist(dict(agg))

    return table


def _extract_total_distribution(
    cells: Dict[int, Dict[int, str]],
    label_aliases: List[str],
    header_candidates: List[str],
) -> Dict[str, float]:
    label_set = {_normalize(x) for x in label_aliases}
    header_set = {_normalize(x) for x in header_candidates}

    label_row = None
    label_col = None
    for row_idx in sorted(cells.keys()):
        row = cells[row_idx]
        for col_idx, val in row.items():
            if _normalize(val) in label_set:
                label_row = row_idx
                label_col = col_idx
                break
        if label_row is not None:
            break

    if label_row is None or label_col is None:
        raise ValueError("Could not locate total row")

    # Find header row above label_row containing known headers
    header_row = None
    for row_idx in range(label_row - 1, 0, -1):
        row = cells.get(row_idx, {})
        hits = sum(1 for v in row.values() if _normalize(v) in header_set)
        if hits >= 3:
            header_row = row_idx
            break

    if header_row is None:
        raise ValueError("Could not locate header row for totals")

    header_cols: Dict[int, str] = {}
    for col_idx, val in cells.get(header_row, {}).items():
        if _normalize(val) in header_set:
            header_cols[col_idx] = str(val).strip()

    if not header_cols:
        raise ValueError("No headers found for totals")

    totals: Dict[str, float] = {}
    total_row = cells.get(label_row, {})
    for col_idx, header in header_cols.items():
        raw = str(total_row.get(col_idx, "")).strip()
        if raw == "":
            continue
        try:
            totals[header] = float(raw) / 100.0
        except ValueError:
            continue

    if not totals:
        raise ValueError("No totals extracted")

    return totals


def _map_distribution(
    raw: Dict[str, float],
    mapping: Dict[str, Optional[str]],
) -> Dict[str, float]:
    agg: Dict[str, float] = defaultdict(float)
    for key, val in raw.items():
        mapped = mapping.get(key)
        if mapped is None:
            continue
        agg[mapped] += val
    return _normalize_dist(dict(agg))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract ethnocultural tables from XLSX")
    parser.add_argument("--input", required=True, help="Path to XLSX")
    parser.add_argument(
        "--output",
        default="config/ethnocultural_tables_2024.json",
        help="Path to output JSON (default: config/ethnocultural_tables_2024.json)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    with zipfile.ZipFile(in_path, "r") as z:
        shared_strings = _load_shared_strings(z)

        # FIGURA 16: language by ethnicity
        sheet16 = _find_sheet_path(z, "Figura 16")
        if not sheet16:
            raise SystemExit("Sheet 'Figura 16' not found")
        cells16 = _load_sheet_cells(z, sheet16, shared_strings)
        headers16, rows16 = _extract_table(
            cells16,
            header_labels=["Etnii"],
            row_label_aliases=["Etnii", "Etnia", "Etnia declarată", "Etnii declarate"],
        )

        # FIGURA 17: religion by ethnicity
        sheet17 = _find_sheet_path(z, "Figura 17")
        if not sheet17:
            raise SystemExit("Sheet 'Figura 17' not found")
        cells17 = _load_sheet_cells(z, sheet17, shared_strings)
        headers17, rows17 = _extract_table(
            cells17,
            header_labels=["Etnii"],
            row_label_aliases=["Etnii", "Etnia", "Etnia declarată", "Etnii declarate"],
        )

    # Row mappings (ethnicity)
    row_map_raw = {
        "Moldoveni": "Moldovean",
        "Români": "Român",
        "Ucraineni": "Ucrainean",
        "Ruși": "Rus",
        "Găgăuzi": "Găgăuz",
        "Bulgari": "Bulgar",
        "Romi/Țigani": "Rrom",
        "Romi/ Tigani": "Rrom",
        "Romi / Țigani": "Rrom",
        "Romi / Tigani": "Rrom",
        "Alte etnii": "Altele",
        "Alte etnii ": "Altele",
        "Nu au declarat etnia": None,
        "Populația totală": None,
        "Total": None,
    }
    # Allow normalized lookup too
    row_map = dict(row_map_raw)
    row_map.update({_normalize(k): v for k, v in row_map_raw.items()})

    # Column mappings for language
    lang_map_raw = {
        "Moldovenească": "Română",
        "Română": "Română",
        "Ucraineană": "Ucraineană",
        "Rusă": "Rusă",
        "Găgăuză": "Găgăuză",
        "Bulgară": "Bulgară",
        "Romani (Țigănească)": "Rromani",
        "Belorusă": "Alta",
        "Germană": "Alta",
        "Poloneză": "Alta",
        "Altă limbă": "Alta",
        "Nu au declarat limba maternă": None,
    }
    lang_map = dict(lang_map_raw)
    lang_map.update({_normalize(k): v for k, v in lang_map_raw.items()})

    # Column mappings for religion
    rel_map_raw = {
        "Ortodoxă": "Ortodox",
        "Baptistă": "Baptist",
        "Martorii lui Iehova": "Martor al lui Iehova",
        "Penticostală": "Penticostal",
        "Adventistă": "Adventist",
        "Creștină după Evanghelie": "Creștină după Evanghelie",
        "Staroveri (Ortodoxă Rusă de rit vechi)": "Staroveri (Ortodoxă Rusă de rit vechi)",
        "Islam": "Islam",
        "Romano-catolică": "Catolic",
        "Altă religie": "Altă religie",
    }
    rel_map = dict(rel_map_raw)
    rel_map.update({_normalize(k): v for k, v in rel_map_raw.items()})

    language_by_ethnicity = _build_ethnicity_table(headers16, rows16, lang_map, row_map)
    religion_by_ethnicity = _build_ethnicity_table(headers17, rows17, rel_map, row_map)

    # Totals for language and religion (national distributions)
    # Language totals from Figura 8 (profile by territory) - use the "Total" row
    with zipfile.ZipFile(in_path, "r") as z:
        shared_strings = _load_shared_strings(z)
        sheet8 = _find_sheet_path(z, "Figura 8")
        if not sheet8:
            raise SystemExit("Sheet 'Figura 8' not found")
        cells8 = _load_sheet_cells(z, sheet8, shared_strings)
        language_total_raw = _extract_total_distribution(
            cells8,
            label_aliases=["Total"],
            header_candidates=list(lang_map_raw.keys()),
        )

        # Religion totals from Figura 17 (same sheet as ethnicity breakdown)
        sheet17 = _find_sheet_path(z, "Figura 17")
        if not sheet17:
            raise SystemExit("Sheet 'Figura 17' not found")
        cells17 = _load_sheet_cells(z, sheet17, shared_strings)
        religion_total_raw = _extract_total_distribution(
            cells17,
            label_aliases=["Populația totală", "Populatia totală", "Populatia totala", "Populația totală"],
            header_candidates=list(rel_map_raw.keys()),
        )

    # Mapping for totals (Moldovenească -> Română, other non-modeled -> Alta or drop)
    lang_total_map_raw: Dict[str, Optional[str]] = {
        "Moldovenească": "Română",
        "Română": "Română",
        "Ucraineană": "Ucraineană",
        "Rusă": "Rusă",
        "Găgăuză": "Găgăuză",
        "Bulgară": "Bulgară",
        "Romani (Țigănească)": "Rromani",
        "Belorusă": "Alta",
        "Germană": "Alta",
        "Poloneză": "Alta",
        "Altă limbă": "Alta",
        "Nu au declarat limba maternă": None,
    }

    rel_total_map_raw: Dict[str, Optional[str]] = dict(rel_map_raw)

    language_distribution = _map_distribution(language_total_raw, lang_total_map_raw)
    religion_distribution = _map_distribution(religion_total_raw, rel_total_map_raw)

    payload = {
        "source": {
            "file": in_path.name,
            "extracted_on": datetime.now().isoformat(),
            "sheets": {
                "language_by_ethnicity": "Figura 16",
                "religion_by_ethnicity": "Figura 17",
            },
        },
        "mappings": {
            "ethnicity": row_map_raw,
            "language": lang_map_raw,
            "religion": rel_map_raw,
            "language_total": lang_total_map_raw,
            "religion_total": rel_total_map_raw,
        },
        "language_by_ethnicity": language_by_ethnicity,
        "religion_by_ethnicity": religion_by_ethnicity,
        "language_distribution": language_distribution,
        "religion_distribution": religion_distribution,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path}")
    print("Ethnicities:", ", ".join(sorted(language_by_ethnicity.keys())))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
