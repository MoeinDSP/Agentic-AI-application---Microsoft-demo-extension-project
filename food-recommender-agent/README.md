# Agentic-AI-application---Microsoft-demo-extension-project




#### Agent 4 run TEST: 

```
# Unit tests only (no Agent 4 needed)
pytest tests/ -m unit -v
# Integration tests (Agent 4 must be running)
uvicorn app.main:app --port 8004 &
pytest tests/ -m integration -v
# Mock Agent 3 CLI
python -m tests.mock_agent3 --scenario lunch
python -m tests.mock_agent3 --scenario dinner --budget 60 --radius 1200
```