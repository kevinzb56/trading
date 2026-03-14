"""
Layer 5: Entity Relationship Graph Reasoning.

Constructs a graph from extracted entities, events, and sentiment,
then derives features from graph topology for market impact inference.

Graph structure:
  Nodes: entities (people, countries, orgs, commodities, etc.)
  Edges: relationships weighted by co-occurrence, sentiment, event context

Features extracted:
  - Per-asset propagation scores
  - Graph centrality metrics
  - Cluster coefficients
  - Shortest path lengths to asset nodes
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

logger = logging.getLogger(__name__)

# Canonical asset nodes in the graph
ASSET_NODES = {
    "gold": {"type": "asset", "aliases": ["gold", "xauusd", "precious metal", "safe haven"]},
    "equities": {"type": "asset", "aliases": ["equities", "stocks", "s&p", "nasdaq", "dow", "market"]},
    "btc": {"type": "asset", "aliases": ["bitcoin", "btc", "crypto"]},
    "cl": {"type": "asset", "aliases": ["crude oil", "oil", "petroleum", "cl"]},
    "wheat": {"type": "asset", "aliases": ["wheat", "grain", "agriculture"]},
    "eurodollar": {"type": "asset", "aliases": ["eurodollar", "eur/usd", "euro", "dollar", "fx"]},
    "treasury_2y": {"type": "asset", "aliases": ["treasury", "2y yield", "bonds", "rates"]},
}

# Transmission rules: how events propagate through the graph to assets
TRANSMISSION_RULES = {
    # (entity_type, event_type) -> {asset: (direction_multiplier, weight)}
    ("country:china", "trade_war"): {
        "equities": (-1, 0.8), "gold": (1, 0.7), "eurodollar": (-1, 0.5),
        "btc": (0, 0.2), "cl": (-1, 0.3), "wheat": (1, 0.3), "treasury_2y": (-1, 0.4),
    },
    ("country:russia", "geopolitical_conflict"): {
        "gold": (1, 0.8), "cl": (1, 0.7), "wheat": (1, 0.8),
        "equities": (-1, 0.5), "eurodollar": (0, 0.3), "btc": (0, 0.2), "treasury_2y": (-1, 0.4),
    },
    ("country:iran", "sanctions"): {
        "cl": (1, 0.8), "gold": (1, 0.6), "equities": (-1, 0.3),
        "wheat": (0, 0.1), "eurodollar": (0, 0.2), "btc": (0, 0.1), "treasury_2y": (-1, 0.3),
    },
    ("country:iran", "geopolitical_conflict"): {
        "cl": (1, 0.8), "gold": (1, 0.7), "equities": (-1, 0.4),
        "wheat": (0, 0.1), "eurodollar": (0, 0.2), "btc": (0, 0.1), "treasury_2y": (-1, 0.3),
    },
    ("organization:fed", "monetary_policy"): {
        "treasury_2y": (1, 0.9), "equities": (-1, 0.6), "gold": (-1, 0.7),
        "eurodollar": (1, 0.5), "btc": (-1, 0.3), "cl": (0, 0.2), "wheat": (0, 0.1),
    },
    ("*", "trade_war"): {
        "equities": (-1, 0.6), "gold": (1, 0.5), "eurodollar": (-1, 0.3),
        "btc": (0, 0.1), "cl": (-1, 0.2), "wheat": (0, 0.2), "treasury_2y": (-1, 0.3),
    },
    ("*", "fiscal_policy"): {
        "equities": (1, 0.5), "treasury_2y": (1, 0.6), "gold": (1, 0.3),
        "eurodollar": (0, 0.2), "btc": (0, 0.1), "cl": (0, 0.1), "wheat": (0, 0.1),
    },
    ("*", "sanctions"): {
        "gold": (1, 0.5), "cl": (1, 0.4), "equities": (-1, 0.3),
        "eurodollar": (0, 0.2), "btc": (0, 0.1), "wheat": (0, 0.2), "treasury_2y": (-1, 0.2),
    },
    ("*", "geopolitical_conflict"): {
        "gold": (1, 0.7), "cl": (1, 0.4), "equities": (-1, 0.5),
        "eurodollar": (0, 0.2), "btc": (0, 0.1), "wheat": (0, 0.3), "treasury_2y": (-1, 0.4),
    },
    ("*", "military"): {
        "gold": (1, 0.7), "cl": (1, 0.5), "equities": (-1, 0.5),
        "eurodollar": (0, 0.2), "btc": (0, 0.1), "wheat": (0, 0.2), "treasury_2y": (-1, 0.4),
    },
    ("*", "energy_policy"): {
        "cl": (1, 0.7), "gold": (0, 0.2), "equities": (0, 0.3),
        "eurodollar": (0, 0.1), "btc": (0, 0.1), "wheat": (0, 0.1), "treasury_2y": (0, 0.1),
    },
    ("*", "crypto_policy"): {
        "btc": (1, 0.8), "equities": (0, 0.2), "gold": (0, 0.1),
        "cl": (0, 0.0), "wheat": (0, 0.0), "eurodollar": (0, 0.1), "treasury_2y": (0, 0.1),
    },
    ("*", "regulation"): {
        "equities": (-1, 0.4), "btc": (-1, 0.3), "gold": (0, 0.1),
        "cl": (0, 0.1), "wheat": (0, 0.1), "eurodollar": (0, 0.1), "treasury_2y": (0, 0.1),
    },
}


@dataclass
class GraphFeatures:
    """Features extracted from the entity relationship graph."""
    # Per-asset propagation signals
    asset_graph_signal: dict = field(default_factory=dict)  # asset -> float (-1 to 1)
    asset_graph_weight: dict = field(default_factory=dict)  # asset -> float (0 to 1)

    # Graph topology
    num_nodes: int = 0
    num_edges: int = 0
    graph_density: float = 0.0
    avg_clustering: float = 0.0
    num_connected_components: int = 0

    # Entity-level
    max_centrality: float = 0.0
    entity_diversity: int = 0  # number of distinct entity types

    def to_feature_dict(self) -> dict:
        """Convert to flat feature dictionary."""
        features = {
            "graph_num_nodes": self.num_nodes,
            "graph_num_edges": self.num_edges,
            "graph_density": self.graph_density,
            "graph_avg_clustering": self.avg_clustering,
            "graph_components": self.num_connected_components,
            "graph_max_centrality": self.max_centrality,
            "graph_entity_diversity": self.entity_diversity,
        }
        for asset in ASSET_NODES:
            features[f"graph_signal_{asset}"] = self.asset_graph_signal.get(asset, 0.0)
            features[f"graph_weight_{asset}"] = self.asset_graph_weight.get(asset, 0.0)
        return features


class EntityGraph:
    """
    Builds an entity-event-asset graph from extracted features
    and derives market impact signals through graph propagation.
    """

    def __init__(self):
        if nx is None:
            raise ImportError("networkx required: pip install networkx")

    def build_and_reason(
        self,
        entities: list,      # List of Entity objects from NER
        events: list,         # List of (event_type, confidence) from event detector
        sentiment_compound: float,  # -1 to 1 from FinBERT
    ) -> GraphFeatures:
        """
        Build graph and extract features + propagation signals.

        1. Add asset nodes
        2. Add entity nodes (from NER)
        3. Add event nodes
        4. Connect entities to assets via ENTITY_TO_ASSET_MAP
        5. Connect events to assets via TRANSMISSION_RULES
        6. Propagate sentiment-weighted signals
        7. Extract topology features
        """
        G = nx.Graph()

        # Step 1: Add asset nodes
        for asset_name in ASSET_NODES:
            G.add_node(f"asset:{asset_name}", node_type="asset")

        # Step 2: Add entity nodes and connect to assets
        entity_types = set()
        for ent in entities:
            node_id = f"entity:{ent.text.lower()}"
            G.add_node(node_id, node_type="entity", label=ent.label, score=ent.score)
            entity_types.add(ent.label)

            # Connect to assets
            for asset in ent.mapped_assets:
                G.add_edge(node_id, f"asset:{asset}",
                           weight=ent.score, edge_type="entity_asset")

        # Step 3: Add event nodes and connect via transmission rules
        for event_type, event_conf in events:
            if event_type == "no_event":
                continue
            event_node = f"event:{event_type}"
            G.add_node(event_node, node_type="event", confidence=event_conf)

            # Connect entities to events
            for ent in entities:
                ent_node = f"entity:{ent.text.lower()}"
                G.add_edge(ent_node, event_node,
                           weight=min(ent.score, event_conf),
                           edge_type="entity_event")

            # Look up transmission rules
            for ent in entities:
                key = (f"{ent.label}:{ent.text.lower()}", event_type)
                rule = TRANSMISSION_RULES.get(key)
                if rule is None:
                    rule = TRANSMISSION_RULES.get(("*", event_type), {})

                for asset, (direction_mult, weight) in rule.items():
                    if weight > 0:
                        G.add_edge(
                            event_node, f"asset:{asset}",
                            weight=weight * event_conf,
                            direction=direction_mult,
                            edge_type="event_asset",
                        )

            # Also apply wildcard rules if no entity-specific rule found
            wildcard_rule = TRANSMISSION_RULES.get(("*", event_type), {})
            for asset, (direction_mult, weight) in wildcard_rule.items():
                edge_key = (event_node, f"asset:{asset}")
                if not G.has_edge(*edge_key):
                    G.add_edge(
                        event_node, f"asset:{asset}",
                        weight=weight * event_conf * 0.5,
                        direction=direction_mult,
                        edge_type="event_asset_wildcard",
                    )

        # Step 4: Propagate signals to compute per-asset scores
        asset_signals = {}
        asset_weights = {}

        for asset_name in ASSET_NODES:
            asset_node = f"asset:{asset_name}"
            total_signal = 0.0
            total_weight = 0.0

            for neighbor in G.neighbors(asset_node):
                edge_data = G.edges[asset_node, neighbor]
                weight = edge_data.get("weight", 0.0)
                direction = edge_data.get("direction", 0)

                # Apply sentiment modulation
                if direction != 0:
                    # Positive sentiment on negative direction → less bearish
                    signal = direction * weight
                    if sentiment_compound != 0:
                        sentiment_mod = 1.0 + 0.3 * sentiment_compound * direction
                        signal *= max(0.1, min(2.0, sentiment_mod))
                else:
                    signal = 0.0

                total_signal += signal
                total_weight += weight

            # Normalize
            if total_weight > 0:
                asset_signals[asset_name] = np.clip(total_signal / total_weight, -1, 1)
                asset_weights[asset_name] = min(1.0, total_weight)
            else:
                asset_signals[asset_name] = 0.0
                asset_weights[asset_name] = 0.0

        # Step 5: Extract topology features
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G) if num_nodes > 1 else 0.0
        avg_clustering = nx.average_clustering(G) if num_nodes > 2 else 0.0
        components = nx.number_connected_components(G)

        # Centrality
        if num_nodes > 0 and num_edges > 0:
            try:
                centrality = nx.degree_centrality(G)
                max_centrality = max(centrality.values()) if centrality else 0.0
            except Exception:
                max_centrality = 0.0
        else:
            max_centrality = 0.0

        return GraphFeatures(
            asset_graph_signal=asset_signals,
            asset_graph_weight=asset_weights,
            num_nodes=num_nodes,
            num_edges=num_edges,
            graph_density=density,
            avg_clustering=avg_clustering,
            num_connected_components=components,
            max_centrality=max_centrality,
            entity_diversity=len(entity_types),
        )

    def batch_reason(
        self,
        entities_list: list,
        events_list: list,
        sentiments: list,
    ) -> list:
        """Process a batch of tweets."""
        results = []
        for entities, events, sent in zip(entities_list, events_list, sentiments):
            try:
                result = self.build_and_reason(entities, events, sent)
            except Exception as e:
                logger.warning(f"Graph reasoning failed: {e}")
                result = GraphFeatures()
            results.append(result)
        return results
