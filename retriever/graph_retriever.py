"""Graph-based retriever for CONDOR mechanism m3 (Graph RAG).

Builds a lightweight co-occurrence knowledge graph from retrieved documents
and expands queries by traversing entity neighbourhood paths.
"""

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

import requests


class GraphRetriever:
    """Retrieves documents using a local entity-co-occurrence knowledge graph.

    Workflow:
        1. Call the same HTTP retrieval backend as Standard RAG (api_url) to
           get an initial candidate set.
        2. Extract named-entity mentions from the candidates using a simple
           regex heuristic (capitalised n-grams).
        3. Add entity nodes and co-occurrence edges to an in-memory NetworkX
           graph.
        4. For the query's entities, walk 1-hop neighbours and re-rank
           candidates by entity overlap with the neighbourhood.
        5. Return the top-num_results re-ranked documents.

    The graph is stateful across calls within one question episode, allowing
    multi-hop retrieval to accumulate entity context.
    """

    def __init__(self, api_url: str, num_results: int = 5, max_hops: int = 1):
        self.api_url = api_url
        self.num_results = num_results
        self.max_hops = max_hops

        if _NX_AVAILABLE:
            self._graph = nx.Graph()
        else:
            self._graph = None
        self._entity_to_docs: Dict[str, List[str]] = defaultdict(list)

    def _extract_entities(self, text: str) -> Set[str]:
        """Simple heuristic: capitalised word sequences of length 1-4."""
        pattern = r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3})\b'
        matches = re.findall(pattern, text)
        return {m.strip() for m in matches if len(m) > 2}

    def _build_graph(self, documents: List[str]) -> None:
        """Add document entities and their co-occurrence edges to the graph."""
        for doc in documents:
            entities = self._extract_entities(doc)
            for ent in entities:
                self._entity_to_docs[ent].append(doc)
                if self._graph is not None:
                    self._graph.add_node(ent)

            entity_list = list(entities)
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    if self._graph is not None:
                        if self._graph.has_edge(entity_list[i], entity_list[j]):
                            self._graph[entity_list[i]][entity_list[j]]['weight'] += 1
                        else:
                            self._graph.add_edge(entity_list[i], entity_list[j], weight=1)

    def _neighbourhood_entities(self, query_entities: Set[str]) -> Set[str]:
        """Return entities within max_hops of any query entity."""
        if self._graph is None or not _NX_AVAILABLE:
            return query_entities
        neighbours: Set[str] = set(query_entities)
        for _ in range(self.max_hops):
            frontier: Set[str] = set()
            for ent in neighbours:
                if self._graph.has_node(ent):
                    frontier.update(self._graph.neighbors(ent))
            neighbours.update(frontier)
        return neighbours

    def _fetch_raw(self, query: str) -> List[str]:
        """Call the HTTP search API and return a list of document strings."""
        try:
            resp = requests.post(
                self.api_url,
                json={'query': query, 'num_results': self.num_results * 3},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return [str(d) for d in data]
            if isinstance(data, dict):
                return [str(d) for d in data.get('results', data.get('documents', []))]
        except Exception:
            pass
        return []

    def retrieve(self, query: str) -> List[str]:
        """Retrieve and graph-re-rank documents for a query.

        Returns at most num_results document strings.
        """
        raw_docs = self._fetch_raw(query)
        if not raw_docs:
            return []

        self._build_graph(raw_docs)

        query_entities = self._extract_entities(query)
        neighbourhood = self._neighbourhood_entities(query_entities)

        # Score each document by entity overlap with neighbourhood
        scored = []
        for doc in raw_docs:
            doc_ents = self._extract_entities(doc)
            overlap = len(doc_ents & neighbourhood)
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored[: self.num_results]]

    def reset(self) -> None:
        """Clear the in-memory graph (call between questions)."""
        if _NX_AVAILABLE:
            self._graph = nx.Graph()
        self._entity_to_docs.clear()
