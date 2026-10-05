import json

import h5py
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.provenance import verify_native_provenance
from native_result_fixtures import declare_native


OWNER = {"point_id": "point-1", "execution_id": "execution-1", "attempt": 1,
         "plan_fingerprint": "a" * 64}


def optical(path, *, source=None, quality=None):
    with h5py.File(path, "w") as handle:
        declare_native(handle, "qcl-negf-optical-v4", "physics.analysis")
        if source is not None:
            handle["metadata/source_state_receipt_json"] = json.dumps(source)
        if quality is not None:
            handle["metadata/stationary_quality_json"] = json.dumps(quality)


def receipt():
    return {"identity": OWNER, "state_id": "state-1", "state_sequence": 4,
            "commit_sha256": "b" * 64}


def test_optical_unknown_quality_remains_unknown_in_embedded_receipt(tmp_path):
    path = tmp_path / "optical.h5"
    quality = {"iterative_converged": False, "physical_gates_passed": False,
               "discretization_verified": "not_measured", "scientific_accepted": False,
               "experimental_validation": "not_evaluated"}
    optical(path, source=receipt(), quality=quality)
    report = verify_native_provenance(path, OWNER, None)
    assert report["source_state"]["state_sequence"] == 4
    assert report["stationary_quality"] == quality


@pytest.mark.parametrize("change", ["missing_source", "missing_quality", "other_attempt", "false_pass"])
def test_optical_cannot_claim_quality_or_ownership_from_sidecar(tmp_path, change):
    source, quality = receipt(), {"scientific_accepted": False,
                                 "discretization_verified": "not_measured"}
    if change == "missing_source":
        source = None
    elif change == "missing_quality":
        quality = None
    elif change == "other_attempt":
        source["identity"] = {**OWNER, "attempt": 2}
    else:
        quality["scientific_accepted"] = True
    path = tmp_path / "optical.h5"
    optical(path, source=source, quality=quality)
    with pytest.raises(ContractError, match="optical|quality|identity"):
        verify_native_provenance(path, OWNER, None)
