import pytest
from pydantic import ValidationError

from agent3_mcp.models.tools import PlaceDetailsRequest, RouteEstimateRequest


def test_route_estimate_request_requires_mode() -> None:
    with pytest.raises(ValidationError):
        RouteEstimateRequest.model_validate(
            {
                "origin": {"lat": 41.9028, "lng": 12.4964},
                "destination": {"lat": 41.8902, "lng": 12.4922},
                "mode": "",
            }
        )


def test_route_estimate_request_normalizes_mode() -> None:
    request = RouteEstimateRequest.model_validate(
        {
            "origin": {"lat": 41.9028, "lng": 12.4964},
            "destination": {"lat": 41.8902, "lng": 12.4922},
            "mode": " WALKING ",
        }
    )

    assert request.mode == "walking"


def test_route_estimate_request_accepts_bicycling_mode() -> None:
    request = RouteEstimateRequest.model_validate(
        {
            "origin": {"lat": 41.9028, "lng": 12.4964},
            "destination": {"lat": 41.8902, "lng": 12.4922},
            "mode": "bicycling",
        }
    )

    assert request.mode == "bicycling"


def test_route_estimate_request_rejects_unsupported_mode() -> None:
    with pytest.raises(ValidationError):
        RouteEstimateRequest.model_validate(
            {
                "origin": {"lat": 41.9028, "lng": 12.4964},
                "destination": {"lat": 41.8902, "lng": 12.4922},
                "mode": "metro",
            }
        )


def test_place_details_request_requires_ids() -> None:
    with pytest.raises(ValidationError):
        PlaceDetailsRequest.model_validate({"place_ids": []})
