"""
VoiceSentinel AI - Integration Tests for FastAPI Backend REST Endpoints & Security Actions
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import (
    app,
    health_check,
    get_enrolled_profiles,
    enroll_profile,
    get_datasets_summary,
    inspect_sip_filter,
    trigger_security_action,
    EnrollRequest,
    SipFilterRequest,
    SecurityActionRequest
)


def test_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert "pyTorchModelStatus" in res


def test_enroll_and_get_profiles():
    get_res = get_enrolled_profiles()
    assert get_res["success"] is True

    req = EnrollRequest(name="Test Officer", organization="Security Cell", f0Mean=160.0)
    post_res = enroll_profile(req)
    assert post_res["success"] is True


def test_datasets_endpoint():
    res = get_datasets_summary()
    assert res["success"] is True
    assert "totalAudioFiles" in res["data"]
    assert "humanMimicryCount" in res["data"]


def test_sip_filter_endpoint():
    req = SipFilterRequest(sipHeader="SIP/2.0 Via: Asterisk-Spoofed", callerNumber="+91 98765 43210")
    res = inspect_sip_filter(req)
    assert res["success"] is True
    assert res["sipTrustScore"] == 25


def test_security_action_allow():
    req = SecurityActionRequest(
        incidentId="inc_allow_001",
        action="ALLOW_CALL",
        riskScore=12.5,
        classification="GENUINE HUMAN VOICE",
        audioHashSha256="a" * 64
    )
    res = trigger_security_action(req)
    assert res["success"] is True
    assert res["callDecisionState"] == "ALLOW"


def test_security_action_block():
    req = SecurityActionRequest(
        incidentId="inc_block_001",
        action="BLOCK_CALL",
        riskScore=88.5,
        classification="AI VOICE CLONE SUSPECTED",
        audioHashSha256="b" * 64
    )
    res = trigger_security_action(req)
    assert res["success"] is True
    assert res["callDecisionState"] == "BLOCKED"


def test_security_action_report():
    req = SecurityActionRequest(
        incidentId="inc_report_001",
        action="REPORT_INCIDENT",
        riskScore=65.0,
        classification="HUMAN MIMICRY / POSSIBLE IMPERSONATION",
        audioHashSha256="c" * 64
    )
    res = trigger_security_action(req)
    assert res["success"] is True
    assert res["callDecisionState"] == "REPORTED"
