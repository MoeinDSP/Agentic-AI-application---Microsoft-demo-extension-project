from datetime import datetime

from models import (
    ClusteringRequest,
    PlaceCandidate,
    Location,
)
from clustering import cluster_places, trip_day_count


def build_sample_request() -> ClusteringRequest:
    return ClusteringRequest(
        trip_start=datetime(2026, 7, 10, 9, 0),
        trip_end=datetime(2026, 7, 12, 21, 0),
        place_candidates=[
            PlaceCandidate(
                id="1",
                name="Duomo di Milano",
                location=Location(latitude=45.4642, longitude=9.1916),
                estimated_visit_duration_minutes=90,
                rating=4.8,
                priority_score=0.95,
            ),
            PlaceCandidate(
                id="2",
                name="Sforza Castle",
                location=Location(latitude=45.4700, longitude=9.1790),
                estimated_visit_duration_minutes=120,
                rating=4.7,
                priority_score=0.9,
            ),
            PlaceCandidate(
                id="3",
                name="Brera District",
                location=Location(latitude=45.4715, longitude=9.1885),
                estimated_visit_duration_minutes=60,
                rating=4.6,
                priority_score=0.85,
            ),
            PlaceCandidate(
                id="4",
                name="Navigli",
                location=Location(latitude=45.4520, longitude=9.1770),
                estimated_visit_duration_minutes=120,
                rating=4.5,
                priority_score=0.8,
            ),
            PlaceCandidate(
                id="5",
                name="San Siro Stadium",
                location=Location(latitude=45.4781, longitude=9.1240),
                estimated_visit_duration_minutes=90,
                rating=4.6,
                priority_score=0.88,
            ),
        ],
    )


def main():
    request = build_sample_request()

    num_days = trip_day_count(request.trip_start, request.trip_end)

    clusters, warnings = cluster_places(
        places=request.place_candidates,
        num_days=num_days,
    )

    print("\n=== CLUSTERING RESULT ===\n")

    for i, day in enumerate(clusters, start=1):
        print(f"Day {i}:")
        for place in day:
            print(f"  - {place.name}")
        print()

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")


if __name__ == "__main__":
    main()