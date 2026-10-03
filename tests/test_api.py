"""Generic job machinery of api/server.py: auth, one job at a time, stop, disk state.
The site and robot-job routes end to end are in test_robot_pipeline.py."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("fastapi")

EXAMPLE = json.loads((ROOT / "examples/farm-uppsala.robotspec.json").read_text())
SPEC = {**EXAMPLE, "site": {**EXAMPLE["site"], "source": "synthetic", "size_m": 150}}
H = {"Authorization": "Bearer t0k"}


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("FARMPAL_DRY_RUN", "1")
    import api.server as server

    monkeypatch.setattr(server, "JOBS", tmp_path)
    monkeypatch.setattr(server, "SITES", tmp_path / "sites")
    monkeypatch.setattr(server, "TOKEN", "t0k")
    return server


@pytest.fixture()
def client(server):
    from fastapi.testclient import TestClient

    return TestClient(server.app)


def test_auth_required(client):
    assert client.get("/health").json()["ok"]
    assert client.get("/robot-jobs").status_code == 401
    assert client.get("/robot-jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/robot-jobs", headers=H).json() == []


def test_unknown_job_and_bad_input(client):
    assert client.get("/robot-jobs/abc123/eval", headers=H).status_code == 404
    assert client.get("/robot-jobs/..%2Fetc/eval", headers=H).status_code == 404
    assert client.get("/sites/..%2Fetc", headers=H).status_code == 404
    assert client.post("/robot-jobs", headers=H, json={**SPEC, "config": {"nope": 1}}).status_code == 422


def test_job_run_by_hand_reports_disk_state(client, tmp_path):
    """A job run with robot_pipeline.py directly (no events.jsonl) is not 'queued'."""
    job = tmp_path / "byhand"
    job.mkdir()
    (job / "robotspec.json").write_text(json.dumps(SPEC))
    env = {**os.environ, "FARMPAL_DRY_RUN": "1", "FARMPAL_SITES": str(tmp_path / "sites")}
    subprocess.run([sys.executable, "robot_pipeline.py", "--job", str(job), "--only", "site"],
                   cwd=ROOT, env=env, capture_output=True, check=True)

    s = client.get("/robot-jobs/byhand", headers=H).json()
    assert s["stages"]["site"]["status"] == "done" and s["stages"]["demos"]["status"] == "pending"
    assert s["state"] == "partial" and s["error"] is None
    with client.stream("GET", "/robot-jobs/byhand/events", headers=H) as r:  # ends instead of idling forever
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert events[-1]["stage"] == "pipeline" and events[-1]["status"] == "done"

    # A stage that died mid-way (last line "running", no process) is an error.
    (job / "logs").mkdir(exist_ok=True)
    with (job / "logs/demos.log").open("a") as f:
        f.write('\n===== now\n{"stage": "demos", "status": "running", "pct": 40, "msg": "", "ts": 0}\n')
    s = client.get("/robot-jobs/byhand", headers=H).json()
    assert s["state"] == "error" and "resume" in s["error"]


SLOW_PIPELINE = '''import json, sys, time
print(json.dumps({"stage": "site", "status": "running", "pct": 1, "msg": "", "ts": time.time()}), flush=True)
time.sleep(120)
'''


def test_one_job_at_a_time_and_stop(client, server, tmp_path, monkeypatch):
    fake = tmp_path / "fakeroot"
    fake.mkdir()
    (fake / "robot_pipeline.py").write_text(SLOW_PIPELINE)
    monkeypatch.setattr(server, "ROOT", fake)

    first = client.post("/robot-jobs", headers=H, json=SPEC).json()["job_id"]
    r = client.post("/robot-jobs", headers=H, json=SPEC)
    assert r.status_code == 409 and r.json()["running_job"] == first
    assert sorted(p.name for p in tmp_path.iterdir() if (p / "robotspec.json").exists()) == [first]  # no orphan dir
    assert client.post(f"/robot-jobs/{first}/resume", headers=H, json={}).status_code == 409
    assert client.post(f"/robot-jobs/{first}/resume", headers=H, json={"from": "nope"}).status_code == 422

    r = client.post(f"/robot-jobs/{first}/stop", headers=H)
    assert r.status_code == 200 and r.json()["state"] == "stopped", r.text
    assert client.get(f"/robot-jobs/{first}", headers=H).json()["state"] == "stopped"
    with client.stream("GET", f"/robot-jobs/{first}/events", headers=H) as s:
        events = [json.loads(l[6:]) for l in s.iter_lines() if l.startswith("data: ")]
    assert events[-1]["stage"] == "pipeline" and events[-1]["status"] == "stopped"
    assert client.post(f"/robot-jobs/{first}/stop", headers=H).status_code == 409  # nothing left to stop

    second = client.post("/robot-jobs", headers=H, json=SPEC)
    assert second.status_code == 200
    client.post(f"/robot-jobs/{second.json()['job_id']}/stop", headers=H)


def test_stop_a_job_run_by_hand(client, tmp_path):
    job = tmp_path / "byhand2"
    job.mkdir()
    (job / "robotspec.json").write_text(json.dumps(SPEC))
    fake = tmp_path / "robot_pipeline.py"
    fake.write_text(SLOW_PIPELINE)
    proc = subprocess.Popen([sys.executable, str(fake), "--job", str(job)])
    try:
        time.sleep(0.5)
        r = client.post("/robot-jobs", headers=H, json=SPEC)
        assert r.status_code == 409 and r.json()["running_job"] == "byhand2"
        assert client.post("/robot-jobs/byhand2/stop", headers=H).status_code == 200
        assert proc.wait(timeout=10) != 0
        assert client.get("/robot-jobs/byhand2", headers=H).json()["state"] == "stopped"
    finally:
        proc.kill()
