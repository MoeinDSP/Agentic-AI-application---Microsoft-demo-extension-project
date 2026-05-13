"""
Compatibility shim — maps the names agent.py expects
to the functions that exist in a2a_client_helper.py
"""
from app.services.a2a_client_helper import (
    call_a2a_agent,
    extract_text_artifact,
    extract_data_artifact,
    call_place_recommender,
    call_clustering_agent,
    call_food_recommender,
)

# agent.py imports these two names specifically
async def send_a2a_message(base_url: str, payload: str, **kwargs):
    return await call_a2a_agent(base_url, payload, **kwargs)

def extract_text_from_task(task: dict) -> str:
    return extract_text_artifact(task)
