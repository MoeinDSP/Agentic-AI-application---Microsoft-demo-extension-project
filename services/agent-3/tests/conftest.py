import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(autouse=True)
def override_planner_dependency() -> None:
    from agent3.main import app
    from agent3.services.a2a import A2AService, get_a2a_service
    from agent3.services.planner import FixedTravelRouteClient, PlannerService, get_planner_service

    planner = PlannerService(FixedTravelRouteClient(0))
    app.dependency_overrides[get_planner_service] = lambda: planner
    app.dependency_overrides[get_a2a_service] = lambda: A2AService(planner)
    yield
    app.dependency_overrides.clear()
