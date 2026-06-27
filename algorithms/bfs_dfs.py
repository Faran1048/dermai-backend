"""
algorithms/bfs_dfs.py

Functions using KNOWLEDGE_GRAPH from knowledge/graph.py:
- bfs_related(start_class: str, max_hops: int = 2) -> list
- dfs_reasoning(start_class: str, depth: int = 3) -> list
"""

from __future__ import annotations
from collections import deque
from knowledge.graph import KNOWLEDGE_GRAPH

def bfs_related(start_class: str, max_hops: int = 2) -> list[str]:
    """
    Perform BFS starting from start_class in the knowledge graph.
    Returns a list of related conditions within max_hops (excluding start_class).
    """
    if start_class not in KNOWLEDGE_GRAPH:
        return []

    visited = {start_class}
    queue = deque([(start_class, 0)])
    result: list[str] = []

    while queue:
        current, hop = queue.popleft()
        if current != start_class:
            result.append(current)

        if hop >= max_hops:
            continue

        # Enqueue neighbors
        neighbors = KNOWLEDGE_GRAPH[current].get("related", [])
        for nbr in neighbors:
            if nbr not in visited and nbr in KNOWLEDGE_GRAPH:
                visited.add(nbr)
                queue.append((nbr, hop + 1))

    return result

def dfs_reasoning(start_class: str, depth: int = 3) -> list[str]:
    """
    Perform recursive DFS starting from start_class to find a clinical reasoning path.
    Explores up to `depth` levels of depth (where start_class is level 0).
    Returns a list of all visited conditions in traversal order.
    """
    visited_list: list[str] = []
    visited_set: set[str] = set()

    def dfs(current: str, current_depth: int) -> None:
        if current_depth > depth or current in visited_set or current not in KNOWLEDGE_GRAPH:
            return
        
        visited_set.add(current)
        visited_list.append(current)
        
        for neighbor in KNOWLEDGE_GRAPH[current].get("related", []):
            dfs(neighbor, current_depth + 1)

    dfs(start_class, 0)
    return visited_list
