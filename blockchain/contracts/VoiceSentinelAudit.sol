// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title VoiceSentinelAudit
 * @dev Smart Contract for recording privacy-preserving immutable voice impersonation audit logs.
 * Stores SHA-256 audio hashes, risk scores, and model classifications on-chain.
 */
contract VoiceSentinelAudit {
    struct AuditRecord {
        string incidentId;
        uint256 riskScore;       // Scaled 0 to 100
        string classification;   // GENUINE or AI_GENERATED
        string audioHash;        // SHA-256 hex string of audio payload
        uint256 timestamp;       // Block timestamp
        string modelVersion;     // Model version identifier
        address reporter;        // Address submitting the record
    }

    // Mapping from incidentId to AuditRecord
    mapping(string => AuditRecord) private auditRecords;
    
    // List of incident IDs for index iteration
    string[] public incidentIds;

    // Log Event
    event AuditLogged(
        string indexed incidentId,
        string audioHash,
        uint256 riskScore,
        string classification,
        uint256 timestamp
    );

    /**
     * @dev Record a new voice impersonation incident log on-chain.
     */
    function logAudit(
        string memory _incidentId,
        uint256 _riskScore,
        string memory _classification,
        string memory _audioHash,
        string memory _modelVersion
    ) public returns (bool) {
        require(bytes(_incidentId).length > 0, "Incident ID required");
        require(bytes(_audioHash).length > 0, "SHA-256 audio hash required");

        AuditRecord memory record = AuditRecord({
            incidentId: _incidentId,
            riskScore: _riskScore,
            classification: _classification,
            audioHash: _audioHash,
            timestamp: block.timestamp,
            modelVersion: _modelVersion,
            reporter: msg.sender
        });

        auditRecords[_incidentId] = record;
        incidentIds.push(_incidentId);

        emit AuditLogged(_incidentId, _audioHash, _riskScore, _classification, block.timestamp);
        return true;
    }

    /**
     * @dev Retrieve recorded audit log by incident ID.
     */
    function getAuditRecord(string memory _incidentId) public view returns (
        string memory incidentId,
        uint256 riskScore,
        string memory classification,
        string memory audioHash,
        uint256 timestamp,
        string memory modelVersion,
        address reporter
    ) {
        AuditRecord memory rec = auditRecords[_incidentId];
        require(bytes(rec.incidentId).length > 0, "Audit record not found");
        return (
            rec.incidentId,
            rec.riskScore,
            rec.classification,
            rec.audioHash,
            rec.timestamp,
            rec.modelVersion,
            rec.reporter
        );
    }

    /**
     * @dev Returns total number of recorded incidents.
     */
    function getTotalIncidents() public view returns (uint256) {
        return incidentIds.length;
    }
}
