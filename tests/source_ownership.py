"""Independent checkout or installed-wheel owners for manufactured test inputs."""

from importlib import metadata
import os
from pathlib import Path


RESULTS_ROOT = Path(__file__).resolve().parents[1]


def contracts_artifacts_path():
    explicit = os.environ.get("QCL_NEGF_TEST_CONTRACTS_SOURCE")
    sibling = RESULTS_ROOT.parent / "qcl-negf-contracts" / "src"
    if sibling.is_dir():
        if explicit is not None:
            assert Path(explicit).is_absolute(), "Contracts source root must be absolute"
            assert Path(explicit).resolve() == sibling.resolve(), "Workspace tests require sibling Contracts"
        return (sibling / "qcl_negf_contracts" / "artifacts.py").resolve()

    if explicit is not None:
        source = Path(explicit)
        assert source.is_absolute(), "Contracts source root must be absolute"
        expected = source / "qcl_negf_contracts" / "artifacts.py"
        assert expected.is_file(), "Contracts source root must contain artifacts.py"
        return expected.resolve()

    distribution = metadata.distribution("qcl-negf-contracts")
    assert distribution.version == "0.2.0", "Tests require the declared Contracts version"
    files = distribution.files or ()
    owners = [row for row in files if row.as_posix() == "qcl_negf_contracts/artifacts.py"]
    assert len(owners) == 1, "Editable standalone Contracts requires an explicit source root"
    return Path(distribution.locate_file(owners[0])).resolve()


def assert_source_ownership(results_file, contracts_file):
    expected_results = RESULTS_ROOT / "src" / "qcl_negf_results" / "state.py"
    assert Path(results_file).resolve() == expected_results.resolve()
    assert Path(contracts_file).resolve() == contracts_artifacts_path()
