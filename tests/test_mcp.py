import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ci_triage.mcp import client


def test_list_tools_discovers_both_tools():
    names = {t["name"] for t in client.list_tools()}
    assert {"fetch_build_artifact", "lookup_test_owner"} <= names


def test_lookup_test_owner_known_test():
    [result] = client.call_many([("lookup_test_owner", {"test_name": "/tests/network/dns-resolution"})])
    assert result["owner"] == "team-networking"


def test_lookup_test_owner_unknown_test():
    [result] = client.call_many([("lookup_test_owner", {"test_name": "no/such/test"})])
    assert result["owner"] == "unknown"


def test_fetch_build_artifact_valid_run():
    [result] = client.call_many([("fetch_build_artifact", {"run_id": "run-101"})])
    assert result["status"] == "failed"
    assert "build_url" in result


def test_fetch_build_artifact_rejects_malformed_run_id():
    [result] = client.call_many([("fetch_build_artifact", {"run_id": "../../etc/passwd"})])
    assert "error" in result
