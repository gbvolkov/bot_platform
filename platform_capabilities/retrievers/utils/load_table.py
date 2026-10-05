import json
from pathlib import Path
import pandas as pd
import datetime

def get_data_from_json(data: str, space_id: str, article_id: str, article_title: str, rename_map: dict):
    # ---------- helpers ----------
    def extract_text(node):
        """Recursively pull plain text from TipTap/ProseMirror nodes."""
        if isinstance(node, dict):
            t = node.get("type")
            if t == "text":
                return node.get("text", "")
            # hardBreak → newline
            if t == "hardBreak":
                return "\n"
            parts = []
            for child in node.get("content", []):
                parts.append(extract_text(child))
            return "".join(parts)
        elif isinstance(node, list):
            return "".join(extract_text(n) for n in node)
        return ""

    def find_tables(tree):
        """Yield every node where type == 'table'."""
        if isinstance(tree, dict):
            if tree.get("type") == "table":
                yield tree
            for child in tree.get("content", []):
                yield from find_tables(child)
        elif isinstance(tree, list):
            for item in tree:
                yield from find_tables(item)


    tables = list(find_tables(data))
    if not tables:
        raise ValueError("No table nodes found in JSON.")

    table = tables[0]  # take the main one

    rows = table.get("content", [])
    if not rows:
        raise ValueError("Table has no rows.")

    # First row = header
    header_cells = rows[0].get("content", [])
    headers_raw = [extract_text(c) for c in header_cells]

    # Clean headers
    headers = [h.strip().replace("\xa0", " ") for h in headers_raw if h.strip()]

    # Collect data rows
    records = []
    for r in rows[1:]:
        cells = r.get("content", [])
        texts = [extract_text(c).strip().replace("\xa0", " ") for c in cells]
        # Pad/truncate to header length
        texts = texts[:len(headers)] + [""] * max(0, len(headers)-len(texts))
        records.append(dict(zip(headers, texts)))

    df = pd.DataFrame(records)

    df = df.rename(columns=rename_map)
    df[["space_id", "article_id", "article_title"]] = (space_id, article_id, article_title)
    #df["problem_description"] = df["problem_description"] + "\n\nСсылка на статью:https://kb.ileasing.ru/space/{space_id}/article/{article_id}"
    return df

def get_glossary_data(data: dict, space_id: str, article_id: str, article_title: str, rename_map: dict):
    import json
    import pandas as pd
    import re

    rows = []
    current_section = None       # tracks whether we are inside the abbreviations or the terms block

    for node in data.get("content", []):
        # -- 1. detect the two H1 headings that mark our sections –
        if node.get("type") == "heading" and node.get("attrs", {}).get("level") == 1:
            heading_text = "".join(child.get("text", "") for child in node.get("content", []))
            heading_low  = heading_text.lower()
            if "сокращ" in heading_low:      # «Список сокращений»
                current_section = "abbr"
            elif "термин" in heading_low:    # «Термины и определения»
                current_section = "term"
            else:
                current_section = None
            continue                         # go to next node

        # -- 2. pick up glossary lines that live inside one of those sections –
        if node.get("type") == "paragraph" and current_section in ("abbr", "term"):
            plain_text = "".join(child.get("text", "") for child in node.get("content", []))

            # split at the first dash rendered as “ – ” (en-dash) or " - "
            term_and_def = re.split(r"\s[–-]\s", plain_text, maxsplit=1)
            if len(term_and_def) == 2:
                term, definition = (part.strip() for part in term_and_def)
                rows.append(
                    {"section": current_section, "term": term, "definition": definition}
                )

    # -- 3. build dataframe –
    df = pd.DataFrame(rows)
    df = df.rename(columns=rename_map)
    df[["space_id", "article_id", "article_title"]] = (space_id, article_id, article_title)

    return df

def _normalize_value(val, typ):
    if val is None:
        return None

    if typ == "title":
        if isinstance(val, dict):
            return val.get("text") or val.get("title") or json.dumps(val, ensure_ascii=False)
        return val

    if typ == "date":
        if isinstance(val, dict):
            raw = val.get("from") or val.get("start") or val.get("value")
            if raw:
                try:
                    dt = datetime.fromisoformat(raw)
                    return dt.date().isoformat() if val.get("withTime") is False else dt.isoformat()
                except Exception:
                    return raw
            return None
        return val

    if typ == "person":
        if isinstance(val, dict):
            return val.get("fullName") or val.get("name") or json.dumps(val, ensure_ascii=False)
        if isinstance(val, list):
            return ", ".join(v.get("fullName") if isinstance(v, dict) else str(v) for v in val)
        return str(val)

    if isinstance(val, dict):
        if "text" in val: return val["text"]
        if "name" in val: return val["name"]
        return json.dumps(val, ensure_ascii=False)

    if isinstance(val, list):
        return ", ".join(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) for v in val)

    return val

#transforms smart table content into dataframe
def get_st_data(payload):
    """
    Returns a wide DataFrame: one row per content item,
    columns are the name values from schemaProperties.
    """
    schema = payload.get("schemaProperties") or []
    # Preserve schema order
    names_in_order = [p.get("name") for p in schema] + ["id", "st_id"]
    code_to_name = {p["code"]: p.get("name") for p in schema}
    name_to_type = {p.get("name"): p.get("type") for p in schema}
    st_id = payload["container"]["id"]

    rows = []
    for item in payload.get("content") or []:
        article = (item or {}).get("article") or {}
        props_outer = article.get("properties") or {}
        props = props_outer.get("properties") if isinstance(props_outer, dict) else {}

        row = {name: None for name in names_in_order}
        row["id"] = article["id"]
        row["st_id"] = st_id
        for code, raw_val in (props or {}).items():
            name = code_to_name.get(code)
            if not name:
                continue  # ignore fields not in the schema
            typ = name_to_type.get(name)
            row[name] = _normalize_value(raw_val, typ)
        rows.append(row)

    return pd.DataFrame(rows, columns=names_in_order)