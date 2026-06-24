from __future__ import annotations

import random

from schemas import Place


class ClusteringService:
    """Groups places into geographically coherent day-clusters using K-means."""

    @staticmethod
    def cluster(
        places: list[Place],
        num_days: int,
        random_seed: int = 42,
    ) -> list[list[Place]]:
        """
        Group places into `num_days` clusters, one per trip day.

        Uses K-means on raw lat/lng coordinates — valid for city-scale distances
        where 1° lat ≈ 1° lng holds to a good approximation. The number of clusters
        is capped at the number of places. Every place is preserved exactly.
        """
        if not places:
            return []

        k = max(1, min(num_days, len(places)))
        if k == 1:
            return [list(places)]

        coords = [(p.lat, p.lng) for p in places]
        labels = ClusteringService._kmeans(coords, k, random_seed=random_seed)

        clusters: list[list[Place]] = [[] for _ in range(k)]
        for place, label in zip(places, labels):
            clusters[label].append(place)

        return [c for c in clusters if c]

    @staticmethod
    def _kmeans(
        coords: list[tuple[float, float]],
        k: int,
        max_iter: int = 100,
        random_seed: int = 42,
    ) -> list[int]:
        """K-means with k-means++ initialization on (lat, lng) coordinates."""
        rng = random.Random(random_seed)
        n = len(coords)

        centroids = ClusteringService._init_centroids_pp(coords, k, rng)
        labels = [0] * n

        for _ in range(max_iter):
            new_labels = [
                min(range(k), key=lambda j, p=point: ClusteringService._sq_dist(p, centroids[j]))
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
                    avg_lng = sum(p[1] for p in cluster_pts) / len(cluster_pts)
                    new_centroids.append((avg_lat, avg_lng))
                else:
                    new_centroids.append(centroids[j])
            centroids = new_centroids

        return labels

    @staticmethod
    def _init_centroids_pp(
        coords: list[tuple[float, float]],
        k: int,
        rng: random.Random,
    ) -> list[tuple[float, float]]:
        """K-means++ centroid initialization for better cluster quality."""
        centroids = [coords[0]]

        for _ in range(1, k):
            sq_dists = [
                min(ClusteringService._sq_dist(p, c) for c in centroids)
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

    @staticmethod
    def _sq_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
