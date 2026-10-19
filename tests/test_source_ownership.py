"""A module from another checkout must not satisfy the owning-source guard."""

from importlib import metadata
from pathlib import Path

import pytest

import qcl_negf_contracts.artifacts as contracts
import qcl_negf_results.state as state
from source_ownership import assert_source_ownership
import source_ownership


def test_selected_results_and_contracts_sources_are_accepted():
    assert_source_ownership(state.__file__, contracts.__file__)


def test_another_results_checkout_is_rejected(tmp_path):
    other = tmp_path / "other-results" / "src" / "qcl_negf_results" / "state.py"
    other.parent.mkdir(parents=True)
    other.write_bytes(Path(state.__file__).read_bytes())
    with pytest.raises(AssertionError):
        assert_source_ownership(other, contracts.__file__)


def test_shadow_contracts_module_is_rejected(tmp_path):
    shadow = tmp_path / "qcl_negf_contracts" / "artifacts.py"
    shadow.parent.mkdir()
    shadow.write_bytes(Path(contracts.__file__).read_bytes())
    with pytest.raises(AssertionError):
        assert_source_ownership(state.__file__, shadow)


def test_workspace_contracts_cannot_be_redirected(tmp_path, monkeypatch):
    results = tmp_path / "components" / "qcl-negf-results"
    sibling = tmp_path / "components" / "qcl-negf-contracts" / "src"
    sibling.mkdir(parents=True)
    another = tmp_path / "another-contracts" / "src"
    another.mkdir(parents=True)
    for source in [sibling, another]:
        package = source / "qcl_negf_contracts"
        package.mkdir()
        (package / "artifacts.py").write_bytes(Path(contracts.__file__).read_bytes())
    monkeypatch.setattr(source_ownership, "RESULTS_ROOT", results)
    monkeypatch.setenv("QCL_NEGF_TEST_CONTRACTS_SOURCE", str(another))
    with pytest.raises(AssertionError):
        source_ownership.contracts_artifacts_path()


@pytest.fixture
def standalone_distribution_metadata(tmp_path, monkeypatch):
    """Synthetic metadata only: no wheel is built, installed, or imported."""
    monkeypatch.setattr(source_ownership, "RESULTS_ROOT", tmp_path / "standalone-results")
    monkeypatch.delenv("QCL_NEGF_TEST_CONTRACTS_SOURCE", raising=False)
    site_packages = tmp_path / "site-packages"
    artifacts = site_packages / "qcl_negf_contracts" / "artifacts.py"
    artifacts.parent.mkdir(parents=True)
    artifacts.write_text("# Synthetic distribution file; never imported.\n")
    dist_info = site_packages / "qcl_negf_contracts-0.2.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: qcl-negf-contracts\nVersion: 0.2.0\n"
    )
    (dist_info / "RECORD").write_text("qcl_negf_contracts/artifacts.py,,\n")
    queries = []

    def distribution(name):
        queries.append(name)
        assert name == "qcl-negf-contracts"
        return metadata.PathDistribution(dist_info)

    monkeypatch.setattr(source_ownership.metadata, "distribution", distribution)
    return artifacts, dist_info, queries


def test_standalone_distribution_ownership(standalone_distribution_metadata):
    artifacts, _, queries = standalone_distribution_metadata
    assert source_ownership.contracts_artifacts_path() == artifacts.resolve()
    assert queries == ["qcl-negf-contracts"]


def test_wrong_distribution_version_is_rejected(standalone_distribution_metadata):
    _, dist_info, queries = standalone_distribution_metadata
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: qcl-negf-contracts\nVersion: 0.2.1\n"
    )
    with pytest.raises(AssertionError, match="Tests require the declared Contracts version"):
        source_ownership.contracts_artifacts_path()
    assert queries == ["qcl-negf-contracts"]
