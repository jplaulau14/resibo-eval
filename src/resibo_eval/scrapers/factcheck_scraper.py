"""
T055: Scrape PH fact-check sites for RAG corpus.

Targets:
  - Vera Files (verafiles.org)
  - Rappler Newsbreak Fact Check (rappler.com/newsbreak/fact-check/)
  - Tsek.ph

Each site-specific scraper:
  1. Crawls the listing/archive pages to collect article URLs
  2. Fetches each article page
  3. Extracts: claim, verdict, date, body text, URL, source, tags
  4. Outputs JSONL to data/raw/

Usage:
    uv run python -m resibo_eval.scrapers.factcheck_scraper --site verafiles --pages 10
    uv run python -m resibo_eval.scrapers.factcheck_scraper --site rappler --pages 10
    uv run python -m resibo_eval.scrapers.factcheck_scraper --site tsek --pages 10
    uv run python -m resibo_eval.scrapers.factcheck_scraper --all --pages 5
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser

DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "raw"
HEADERS = {
    "User-Agent": "Resibo-FactCheck-Scraper/0.1 (hackathon; contact: prod.patricklaurel@gmail.com)"
}
DELAY = 1.5  # seconds between requests — be polite


@dataclass
class FactCheckClaim:
    claim: str
    verdict: str
    date: str
    body: str
    url: str
    source: str
    tags: list[str] = field(default_factory=list)


# ─── Vera Files ──────────────────────────────────────────────────────────────


def scrape_verafiles(max_pages: int = 10) -> list[FactCheckClaim]:
    """Scrape verafiles.org fact-check articles."""
    client = httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30)
    article_urls: list[str] = []

    # Crawl listing pages
    for page in range(1, max_pages + 1):
        url = f"https://verafiles.org/specials/fact-check?page={page}"
        print(f"  [verafiles] listing page {page}: {url}")
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                print(
                    f"  [verafiles] page {page} returned {resp.status_code}, stopping"
                )
                break
            tree = HTMLParser(resp.text)
            links = tree.css("a[href*='/articles/']")
            new_urls = []
            for link in links:
                href = link.attributes.get("href", "")
                if "/articles/" in href and "fact-check" in href.lower():
                    full = (
                        href
                        if href.startswith("http")
                        else f"https://verafiles.org{href}"
                    )
                    if full not in article_urls and full not in new_urls:
                        new_urls.append(full)
            if not new_urls:
                # Try broader match
                links = tree.css("a[href*='/articles/']")
                for link in links:
                    href = link.attributes.get("href", "")
                    if "/articles/" in href:
                        full = (
                            href
                            if href.startswith("http")
                            else f"https://verafiles.org{href}"
                        )
                        if full not in article_urls and full not in new_urls:
                            new_urls.append(full)
            article_urls.extend(new_urls)
            print(
                f"  [verafiles] found {len(new_urls)} articles (total: {len(article_urls)})"
            )
            time.sleep(DELAY)
        except Exception as e:
            print(f"  [verafiles] error on page {page}: {e}")
            break

    # Fetch individual articles
    claims: list[FactCheckClaim] = []
    for i, url in enumerate(article_urls):
        print(f"  [verafiles] article {i+1}/{len(article_urls)}: {url}")
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                continue
            claim = _parse_verafiles_article(resp.text, url)
            if claim:
                claims.append(claim)
        except Exception as e:
            print(f"  [verafiles] error: {e}")
        time.sleep(DELAY)

    client.close()
    return claims


def _parse_verafiles_article(html: str, url: str) -> FactCheckClaim | None:
    tree = HTMLParser(html)

    title_el = tree.css_first("h1, .article-title, .entry-title")
    title = title_el.text(strip=True) if title_el else ""

    # Extract verdict from title (common patterns: FACT CHECK:, FALSE:, etc.)
    verdict = "unknown"
    title_upper = title.upper()
    for v in [
        "FALSE",
        "FAKE",
        "MISLEADING",
        "NEEDS CONTEXT",
        "TRUE",
        "PARTLY FALSE",
        "MISSING CONTEXT",
    ]:
        if v in title_upper:
            verdict = v.lower()
            break

    # Date
    date_el = tree.css_first("time, .date, .published-date, .article-date")
    date_text = ""
    if date_el:
        date_text = date_el.attributes.get("datetime", "") or date_el.text(strip=True)

    # Body text
    body_el = tree.css_first("article, .article-body, .entry-content, .post-content")
    body = ""
    if body_el:
        paragraphs = body_el.css("p")
        body = "\n".join(p.text(strip=True) for p in paragraphs if p.text(strip=True))

    # Tags
    tag_els = tree.css(".tag, .category, a[rel='tag']")
    tags = [t.text(strip=True) for t in tag_els if t.text(strip=True)]

    if not title and not body:
        return None

    # Claim: try to extract from title (remove "FACT CHECK:" prefix)
    claim = title
    for prefix in [
        "FACT CHECK:",
        "VERA FILES FACT CHECK:",
        "FACT-CHECK:",
        "FACT CHECK -",
    ]:
        if claim.upper().startswith(prefix):
            claim = claim[len(prefix) :].strip()
            break

    return FactCheckClaim(
        claim=claim,
        verdict=verdict,
        date=date_text,
        body=body[:3000],  # cap body length for RAG
        url=url,
        source="verafiles",
        tags=tags[:10],
    )


# ─── Rappler ─────────────────────────────────────────────────────────────────


def scrape_rappler(max_pages: int = 10) -> list[FactCheckClaim]:
    """Scrape rappler.com fact-check articles."""
    client = httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30)
    article_urls: list[str] = []

    for page in range(1, max_pages + 1):
        url = f"https://www.rappler.com/newsbreak/fact-check/page/{page}/"
        print(f"  [rappler] listing page {page}: {url}")
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                print(f"  [rappler] page {page} returned {resp.status_code}, stopping")
                break
            tree = HTMLParser(resp.text)
            links = tree.css("a[href*='/newsbreak/fact-check/']")
            new_urls = []
            for link in links:
                href = link.attributes.get("href", "")
                if "/newsbreak/fact-check/" in href and href.count("/") > 4:
                    full = (
                        href
                        if href.startswith("http")
                        else f"https://www.rappler.com{href}"
                    )
                    if (
                        full not in article_urls
                        and full not in new_urls
                        and "/page/" not in full
                    ):
                        new_urls.append(full)
            article_urls.extend(new_urls)
            print(
                f"  [rappler] found {len(new_urls)} articles (total: {len(article_urls)})"
            )
            time.sleep(DELAY)
        except Exception as e:
            print(f"  [rappler] error on page {page}: {e}")
            break

    claims: list[FactCheckClaim] = []
    for i, url in enumerate(article_urls):
        print(f"  [rappler] article {i+1}/{len(article_urls)}: {url}")
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                continue
            claim = _parse_rappler_article(resp.text, url)
            if claim:
                claims.append(claim)
        except Exception as e:
            print(f"  [rappler] error: {e}")
        time.sleep(DELAY)

    client.close()
    return claims


def _parse_rappler_article(html: str, url: str) -> FactCheckClaim | None:
    tree = HTMLParser(html)

    title_el = tree.css_first("h1")
    title = title_el.text(strip=True) if title_el else ""

    verdict = "unknown"
    title_upper = title.upper()
    for v in [
        "FALSE",
        "FAKE",
        "MISLEADING",
        "NEEDS CONTEXT",
        "TRUE",
        "PARTLY FALSE",
        "MISSING CONTEXT",
        "NOT TRUE",
    ]:
        if v in title_upper:
            verdict = v.lower()
            break

    date_el = tree.css_first("time, .posted-on time")
    date_text = ""
    if date_el:
        date_text = date_el.attributes.get("datetime", "") or date_el.text(strip=True)

    body_el = tree.css_first(
        ".post-single__content, article .entry-content, .cXenseParse"
    )
    body = ""
    if body_el:
        paragraphs = body_el.css("p")
        body = "\n".join(p.text(strip=True) for p in paragraphs if p.text(strip=True))

    tag_els = tree.css(".post-single__tag a, a[rel='tag']")
    tags = [t.text(strip=True) for t in tag_els if t.text(strip=True)]

    if not title:
        return None

    claim = title
    for prefix in [
        "FALSE:",
        "MISLEADING:",
        "FACT CHECK:",
        "NOT TRUE:",
        "NEEDS CONTEXT:",
    ]:
        if claim.upper().startswith(prefix):
            claim = claim[len(prefix) :].strip()
            break

    return FactCheckClaim(
        claim=claim,
        verdict=verdict,
        date=date_text,
        body=body[:3000],
        url=url,
        source="rappler",
        tags=tags[:10],
    )


# ─── Tsek.ph ────────────────────────────────────────────────────────────────


def scrape_tsek(max_pages: int = 10) -> list[FactCheckClaim]:
    """Scrape tsek.ph fact-check articles."""
    client = httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30)
    article_urls: list[str] = []

    for page in range(1, max_pages + 1):
        url = f"https://www.tsek.ph/page/{page}/"
        print(f"  [tsek] listing page {page}: {url}")
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                print(f"  [tsek] page {page} returned {resp.status_code}, stopping")
                break
            tree = HTMLParser(resp.text)
            links = tree.css("a[href*='tsek.ph/']")
            new_urls = []
            for link in links:
                href = link.attributes.get("href", "")
                if (
                    "tsek.ph/" in href
                    and "/page/" not in href
                    and "/category/" not in href
                ):
                    # Filter to article pages (not home, not categories)
                    parts = href.rstrip("/").split("/")
                    slug = parts[-1] if parts else ""
                    if (
                        slug
                        and slug not in ("", "www.tsek.ph", "tsek.ph")
                        and not slug.startswith("#")
                    ):
                        full = (
                            href
                            if href.startswith("http")
                            else f"https://www.tsek.ph{href}"
                        )
                        if full not in article_urls and full not in new_urls:
                            new_urls.append(full)
            article_urls.extend(new_urls)
            print(
                f"  [tsek] found {len(new_urls)} articles (total: {len(article_urls)})"
            )
            time.sleep(DELAY)
        except Exception as e:
            print(f"  [tsek] error on page {page}: {e}")
            break

    claims: list[FactCheckClaim] = []
    for i, url in enumerate(article_urls):
        print(f"  [tsek] article {i+1}/{len(article_urls)}: {url}")
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                continue
            claim = _parse_tsek_article(resp.text, url)
            if claim:
                claims.append(claim)
        except Exception as e:
            print(f"  [tsek] error: {e}")
        time.sleep(DELAY)

    client.close()
    return claims


def _parse_tsek_article(html: str, url: str) -> FactCheckClaim | None:
    tree = HTMLParser(html)

    title_el = tree.css_first("h1, .entry-title, .post-title")
    title = title_el.text(strip=True) if title_el else ""

    verdict = "unknown"
    title_upper = title.upper()
    for v in [
        "FALSE",
        "FAKE",
        "MISLEADING",
        "NEEDS CONTEXT",
        "TRUE",
        "PARTLY FALSE",
        "NOT TRUE",
        "MISSING CONTEXT",
    ]:
        if v in title_upper:
            verdict = v.lower()
            break

    date_el = tree.css_first("time, .entry-date, .published")
    date_text = ""
    if date_el:
        date_text = date_el.attributes.get("datetime", "") or date_el.text(strip=True)

    body_el = tree.css_first(".entry-content, article, .post-content")
    body = ""
    if body_el:
        paragraphs = body_el.css("p")
        body = "\n".join(p.text(strip=True) for p in paragraphs if p.text(strip=True))

    tag_els = tree.css("a[rel='tag'], .tag-link")
    tags = [t.text(strip=True) for t in tag_els if t.text(strip=True)]

    if not title:
        return None

    return FactCheckClaim(
        claim=title,
        verdict=verdict,
        date=date_text,
        body=body[:3000],
        url=url,
        source="tsek",
        tags=tags[:10],
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────


def save_claims(claims: list[FactCheckClaim], filename: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for claim in claims:
            f.write(json.dumps(asdict(claim), ensure_ascii=False) + "\n")
    print(f"Saved {len(claims)} claims to {path}")
    return path


SCRAPERS = {
    "verafiles": ("verafiles_claims.jsonl", scrape_verafiles),
    "rappler": ("rappler_claims.jsonl", scrape_rappler),
    "tsek": ("tsek_claims.jsonl", scrape_tsek),
}


def main():
    parser = argparse.ArgumentParser(description="Scrape PH fact-check sites")
    parser.add_argument("--site", choices=list(SCRAPERS.keys()), help="Site to scrape")
    parser.add_argument("--all", action="store_true", help="Scrape all sites")
    parser.add_argument(
        "--pages", type=int, default=5, help="Max listing pages per site"
    )
    args = parser.parse_args()

    if args.all:
        sites = list(SCRAPERS.keys())
    elif args.site:
        sites = [args.site]
    else:
        parser.error("Specify --site or --all")
        return

    total = 0
    for site in sites:
        filename, scraper_fn = SCRAPERS[site]
        print(f"\n{'='*60}")
        print(f"Scraping {site} (max {args.pages} pages)")
        print(f"{'='*60}")
        claims = scraper_fn(max_pages=args.pages)
        save_claims(claims, filename)
        total += len(claims)

    print(f"\n{'='*60}")
    print(f"Total: {total} claims scraped")


if __name__ == "__main__":
    main()
