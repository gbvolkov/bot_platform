"""Command-line helper that searches vacancies on hh.ru via the public REST API."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from typing import Any, Dict, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import requests

import config

from transliterate import translit
from natasha import MorphVocab
import pymorphy2


def _transliterate_latin_to_cyrillic(text: str) -> str:
    """Transliterate Latin names to Cyrillic using transliterate library."""
    try:
        return translit(text, 'ru')
    except Exception:
        return text.lower()

MORPH_ANALYZER = None
MORPH_VOCAB = None


def _get_morph():
    global MORPH_ANALYZER
    if MORPH_ANALYZER is None:
        MORPH_ANALYZER = pymorphy2.MorphAnalyzer()
    return MORPH_ANALYZER


def _get_morph_vocab():
    global MORPH_VOCAB
    if MORPH_VOCAB is None:
        MORPH_VOCAB = MorphVocab()
    return MORPH_VOCAB



API_BASE = "https://api.hh.ru"
HEADERS = {
    "User-Agent": "findjob-bot/1.0 (+https://github.com/your-org/findjob)",
    "Accept": "application/json",
}



def get_token()-> str:
    """
    Request an application access_token via OAuth2 client credentials.
    Reissuing sooner than 5 minutes is avoided to prevent revoking the current token.
    """
    data = {
        "grant_type": "authorization_code",  # standard OAuth2 for app token
        "client_id": config.CLIENT_ID,
        "client_secret": config.CLIENT_SECRET,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    resp = requests.post(config.HH_TOKEN_URL, data=data, headers=headers, timeout=20)
    if resp.status_code == 200:
        payload = resp.json()
        return payload["access_token"]

    # Helpful errors
    if resp.status_code == 400:
        raise RuntimeError(
            f"Bad request to HH token endpoint (400). "
            f"Check grant_type/client_id/client_secret formatting. Response: {resp.text}"
        )
    if resp.status_code == 403:
        raise RuntimeError(
            f"Forbidden (403). The app may not be allowed to get a token. Response: {resp.text}"
        )

    resp.raise_for_status()

#OAUTH_TOKEN = get_token()


def _normalize_region_name(region_name: str) -> str:
    """Prepare a region string for HH suggests lookup (Cyrillic root)."""
    if not region_name:
        return ''
    region = region_name.strip().lower()
    for sep in (',', '/', '(', ')'):
        if sep in region:
            region = region.split(sep, 1)[0]
            break
    region = region.strip()

    latin_aliases = {
        'moscow': 'москва',
        'moskva': 'москва',
        'saint petersburg': 'санкт-петербург',
        'st petersburg': 'санкт-петербург',
        'st. petersburg': 'санкт-петербург',
        'petersburg': 'петербург',
        'novosibirsk': 'новосибирск',
        'yekaterinburg': 'екатеринбург',
    }
    if region in latin_aliases:
        region = latin_aliases[region]
    elif any('a' <= ch <= 'z' for ch in region):
        region = _transliterate_latin_to_cyrillic(region)

    tokens = [token for token in re.findall(r'[а-яё0-9-]+', region) if token]
    if not tokens:
        return region_name.strip().lower()

    morph = _get_morph()
    morph_vocab = _get_morph_vocab()
    lemmas = []
    for token in tokens:
        if parsed := morph.parse(token):
            lemma = parsed[0].normal_form
            lemmas.append(lemma)
    if lemmas:
        normalized = ' '.join(lemmas)
        if hasattr(morph_vocab, 'normalize'):
            normalized = morph_vocab.normalize(normalized.capitalize()).lower()
    else:
        normalized = ' '.join(tokens)

    root = normalized.split()[0]
    for suffix in ('ов', 'ев', 'ёв', 'ий', 'ый', 'ая', 'яя', 'ое', 'ее', 'а'):
        if root.endswith(suffix) and len(root) > len(suffix) + 1:
            root = root[:-len(suffix)]
            break
    while len(root) > 3 and root[-1] in 'аяыуеоиьйъэ':
        root = root[:-1]
    return root or normalized

def fetch_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Perform a GET request expecting a JSON response."""
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def resolve_area_id(region_name: str) -> str:
    """Resolve human-readable region name to hh.ru area identifier."""
    query = _normalize_region_name(region_name)
    if not query:
        raise ValueError("Region name is empty.")
    params = {"text": query, "locale": "RU"}
    data = fetch_json(f"{API_BASE}/suggests/areas", params)
    items = data.get("items", [])
    if not items:
        raise ValueError(f"Не удалось определить регион по запросу '{region_name}'.")
    normalized_query = query.lower().strip()
    for item in items:
        text_value = (item.get("text") or "").lower().strip()
        if text_value == normalized_query:
            return item["id"]
        cleaned = _normalize_region_name(text_value)
        if cleaned == normalized_query:
            return item["id"]
    return items[0]["id"]


def search_vacancies(
    text: str,
    area_id: str | None,
    salary_from: int | None,
    per_page: int,
) -> Iterable[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "per_page": per_page,
        "order_by": "relevance",
    }
    if text:
        params["text"] = text
    if area_id:
        params["area"] = area_id
    if salary_from:
        params["salary"] = salary_from
        params["only_with_salary"] = "true"
    data = fetch_json(f"{API_BASE}/vacancies", params)
    return data.get("items", [])


def format_salary(info: Dict[str, Any] | None) -> str:
    if not info:
        return "зарплата не указана"
    salary_from = info.get("from")
    salary_to = info.get("to")
    currency = info.get("currency") or ""
    parts = []
    if salary_from:
        parts.append(f"?? {salary_from:,}".replace(",", " "))
    if salary_to:
        parts.append(f"?? {salary_to:,}".replace(",", " "))
    if not parts:
        return "зарплата не указана"
    return f"{' '.join(parts)} {currency}".strip()


def run_cli(text: str, salary_from: int | None, region: str, per_page: int) -> int:
    try:
        area_id = resolve_area_id(region) if region else None
    except Exception as exc:  # pragma: no cover - network failure surfaced to user
        print(f"Ошибка определения региона: {exc}", file=sys.stderr)
        return 1

    try:
        vacancies = list(
            search_vacancies(text=text, area_id=area_id, salary_from=salary_from, per_page=per_page)
        )
    except requests.HTTPError as exc:
        print(
            f"Ошибка запроса вакансий: {exc.response.status_code} {exc.response.text}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # pragma: no cover - network failure
        print(f"Ошибка поиска вакансий: {exc}", file=sys.stderr)
        return 1

    if not vacancies:
        print("Вакансии не найдены.")
        return 0

    print(f"Найденные вакансии (всего {len(vacancies)}):")
    print("-" * 80)
    for item in vacancies:
        title = item.get("name", "Без названия")
        employer = (item.get("employer") or {}).get("name", "Не указан")
        salary = format_salary(item.get("salary"))
        url = item.get("alternate_url") or item.get("url")
        print(f"{title} — {employer}")
        print(f"   {salary}")
        if url:
            print(f"   {url}")
        print("-" * 80)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Поиск вакансий на hh.ru через публичный API",
    )
    parser.add_argument(
        "--profession",
        dest="text",
        default="QA Engineer",
        help="Название профессии (поисковый запрос).",
    )
    parser.add_argument(
        "--salary-from",
        dest="salary_from",
        type=int,
        default=None,
        help="Минимальная зарплата (руб.).",
    )
    parser.add_argument(
        "--region",
        default="Москва",
        help="Регион поиска (на русском).",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=20,
        help="Количество вакансий на страницу (1-100).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_cli(
        text=args.text,
        salary_from=args.salary_from,
        region=args.region,
        per_page=max(1, min(args.per_page, 100)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
