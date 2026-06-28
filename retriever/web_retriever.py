"""Web-search retriever for CONDOR mechanism m4 (Web Search RAG).

Uses the DuckDuckGo Lite search endpoint (no API key required) to fetch
live web snippets.  Falls back to a Google Custom Search API if configured.
"""

import re
from typing import List, Optional

import requests


_DDG_URL = "https://html.duckduckgo.com/html/"
_GOOGLE_API_URL = "https://www.googleapis.com/customsearch/v1"


class WebRetriever:
    """Retrieve live web snippets for a query.

    Attempts DuckDuckGo Lite first; falls back to Google CSE if
    google_api_key and google_cx are provided.

    Returned documents are cleaned snippet strings suitable for downstream
    DocumentSelectionAgent and AnswerGenerationAgent.
    """

    def __init__(
        self,
        num_results: int = 5,
        google_api_key: Optional[str] = None,
        google_cx: Optional[str] = None,
        timeout: int = 15,
    ):
        self.num_results = num_results
        self.google_api_key = google_api_key
        self.google_cx = google_cx
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[str]:
        """Return a list of up to num_results snippet strings."""
        docs = self._ddg_search(query)
        if not docs and self.google_api_key and self.google_cx:
            docs = self._google_search(query)
        return docs[: self.num_results]

    # ------------------------------------------------------------------
    # DuckDuckGo Lite scraper (no API key)
    # ------------------------------------------------------------------

    def _ddg_search(self, query: str) -> List[str]:
        try:
            resp = requests.post(
                _DDG_URL,
                data={"q": query, "b": "", "kl": "us-en"},
                headers={"User-Agent": "Mozilla/5.0 (CONDOR-WebRetriever/1.0)"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return self._parse_ddg_html(resp.text)
        except Exception:
            return []

    def _parse_ddg_html(self, html: str) -> List[str]:
        # Extract text snippets from DuckDuckGo HTML result blobs
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )
        clean = []
        for s in snippets:
            text = re.sub(r'<[^>]+>', '', s).strip()
            text = re.sub(r'\s+', ' ', text)
            if text:
                clean.append(text)
        return clean

    # ------------------------------------------------------------------
    # Google Custom Search API
    # ------------------------------------------------------------------

    def _google_search(self, query: str) -> List[str]:
        try:
            params = {
                "key": self.google_api_key,
                "cx": self.google_cx,
                "q": query,
                "num": self.num_results,
            }
            resp = requests.get(_GOOGLE_API_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            snippets = []
            for item in items:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                combined = f"{title}. {snippet}".strip()
                if combined:
                    snippets.append(combined)
            return snippets
        except Exception:
            return []
