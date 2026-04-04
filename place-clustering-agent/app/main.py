from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from clustering import cluster_places, trip_day_count
from models import ClusterDay, ClusteringRequest, ClusteringResult

app = FastAPI(
    title="Place Clustering Agent",
    version="0.1.0",
    description="Clusters recommended places into geographically coherent trip days."
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/cluster", response_model=ClusteringResult)
def cluster_endpoint(request: ClusteringRequest):
    num_days = trip_day_count(request.trip_start, request.trip_end)

    clustered_places, warnings = cluster_places(
        places=request.place_candidates,
        num_days=num_days,
    )

    cluster_days = []
    for idx, places in enumerate(clustered_places, start=1):
        total_minutes = sum(p.estimated_visit_duration_minutes or 60 for p in places)
        cluster_days.append(
            ClusterDay(
                day_index=idx,
                total_estimated_visit_minutes=total_minutes,
                places=places,
            )
        )

    result = ClusteringResult(
        clustered_place_candidates=clustered_places,
        cluster_days=cluster_days,
        warnings=warnings,
    )
    return JSONResponse(content=result.model_dump(mode="json"))