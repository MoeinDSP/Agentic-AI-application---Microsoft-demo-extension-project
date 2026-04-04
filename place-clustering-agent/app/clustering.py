from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt
from typing import List

from models import PlaceCandidate


EARTH_RADIUS_KM = 6371.0
DEFAULT_DAY_CAPACITY_MINUTES = 8 * 60


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c


@dataclass
class WorkingCluster:
    places: List[PlaceCandidate]
    total_minutes: int

    @property
    def centroid(self) -> tuple[float, float]:
        if not self.places:
            return (0.0, 0.0)
        lat = sum(p.location.latitude for p in self.places) / len(self.places)
        lon = sum(p.location.longitude for p in self.places) / len(self.places)
        return lat, lon


def trip_day_count(trip_start, trip_end) -> int:
    # inclusive day count
    return max(1, (trip_end.date() - trip_start.date()).days + 1)


def sort_places_for_assignment(places: List[PlaceCandidate]) -> List[PlaceCandidate]:
    # Higher priority first, then longer visits first
    return sorted(
        places,
        key=lambda p: (
            -(p.priority_score or 0.0),
            -p.estimated_visit_duration_minutes,
            -(p.rating or 0.0),
        ),
    )


def distance_to_cluster(place: PlaceCandidate, cluster: WorkingCluster) -> float:
    c_lat, c_lon = cluster.centroid
    return haversine_km(
        place.location.latitude,
        place.location.longitude,
        c_lat,
        c_lon,
    )


def cluster_places(
    places: List[PlaceCandidate],
    num_days: int,
    day_capacity_minutes: int = DEFAULT_DAY_CAPACITY_MINUTES,
) -> tuple[List[List[PlaceCandidate]], List[str]]:
    warnings: List[str] = []

    if not places:
        return [[] for _ in range(num_days)], warnings

    clusters = [WorkingCluster(places=[], total_minutes=0) for _ in range(num_days)]
    ordered_places = sort_places_for_assignment(places)

    for place in ordered_places:
        place_minutes = place.estimated_visit_duration_minutes or 60

        # Prefer clusters that are geographically close and not overloaded
        ranked_clusters = sorted(
            range(len(clusters)),
            key=lambda idx: (
                clusters[idx].total_minutes >= day_capacity_minutes,
                distance_to_cluster(place, clusters[idx]) if clusters[idx].places else 0.0,
                clusters[idx].total_minutes,
            ),
        )

        chosen_idx = ranked_clusters[0]
        clusters[chosen_idx].places.append(place)
        clusters[chosen_idx].total_minutes += place_minutes

    # Rebalance empty clusters if trip has more days than dense clusters
    for idx, cluster in enumerate(clusters):
        if cluster.places:
            continue

        donor_idx = max(
            range(len(clusters)),
            key=lambda i: len(clusters[i].places)
        )
        donor = clusters[donor_idx]
        if len(donor.places) > 1:
            moved = donor.places.pop()
            donor.total_minutes -= moved.estimated_visit_duration_minutes or 60
            cluster.places.append(moved)
            cluster.total_minutes += moved.estimated_visit_duration_minutes or 60

    for i, cluster in enumerate(clusters, start=1):
        if cluster.total_minutes > day_capacity_minutes:
            warnings.append(
                f"Day {i} may be overloaded: estimated {cluster.total_minutes} minutes."
            )

    return [cluster.places for cluster in clusters], warnings