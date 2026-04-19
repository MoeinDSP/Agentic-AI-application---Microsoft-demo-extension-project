from __future__ import annotations

import random
from math import ceil
from datetime import datetime

from app.models import PlaceCandidate


def cluster_places(
    place_candidates: list[PlaceCandidate],
    trip_start: datetime,
    trip_end: datetime,
    random_seed: int = 42,
) -> list[list[PlaceCandidate]]:
    """
    Group places into geographically coherent clusters using K-means.

    Number of clusters equals trip duration in days (one cluster per day).
    Uses lat/lon coordinates — valid for city-scale distances where
    1° lat ≈ 1° lon holds to a good approximation.
    """
    if not place_candidates:
        return []

    delta = trip_end - trip_start
    num_days = max(1, ceil(delta.total_seconds() / 86400))
    k = min(num_days, len(place_candidates))

    if k == 1:
        return [list(place_candidates)]

    coords = [(p.location.latitude, p.location.longitude) for p in place_candidates]
    labels = _kmeans(coords, k, random_seed=random_seed)

    clusters: list[list[PlaceCandidate]] = [[] for _ in range(k)]
    for place, label in zip(place_candidates, labels):
        clusters[label].append(place)

    return [c for c in clusters if c]


def _kmeans(
    coords: list[tuple[float, float]],
    k: int,
    max_iter: int = 100,
    random_seed: int = 42,
) -> list[int]:
    """K-means with k-means++ initialization on (lat, lon) coordinates."""
    rng = random.Random(random_seed)
    n = len(coords)

    centroids = _init_centroids_pp(coords, k, rng)
    labels = [0] * n

    for _ in range(max_iter):
        new_labels = [
            min(range(k), key=lambda j, p=point: _sq_dist(p, centroids[j]))
            for point in coords
        ]

        if new_labels == labels:
            break
        labels = new_labels

        new_centroids = []
        for j in range(k):
            cluster_pts = [coords[i] for i in range(n) if labels[i] == j]
            if cluster_pts:
                avg_lat = sum(p[0] for p in cluster_pts) / len(cluster_pts)
                avg_lon = sum(p[1] for p in cluster_pts) / len(cluster_pts)
                new_centroids.append((avg_lat, avg_lon))
            else:
                new_centroids.append(centroids[j])
        centroids = new_centroids

    return labels


def _init_centroids_pp(
    coords: list[tuple[float, float]],
    k: int,
    rng: random.Random,
) -> list[tuple[float, float]]:
    """K-means++ centroid initialization for better cluster quality."""
    centroids = [coords[0]]

    for _ in range(1, k):
        sq_dists = [
            min(_sq_dist(p, c) for c in centroids)
            for p in coords
        ]
        total = sum(sq_dists)
        if total == 0:
            centroids.append(coords[len(centroids) % len(coords)])
            continue

        threshold = rng.random() * total
        cumulative = 0.0
        chosen = coords[-1]
        for i, d in enumerate(sq_dists):
            cumulative += d
            if cumulative >= threshold:
                chosen = coords[i]
                break
        centroids.append(chosen)

    return centroids


def _sq_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
