"""One native5 provider context, canonical cursor and closed same-open capture."""
import copy
import hashlib
import pytest
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.state import StateReader
from qcl_negf_results.live_views import describe_live_history, read_live_history
from result_v2_fixture import IDENTITY, source, record


def test_provider_description_does_not_open_hdf_and_records_preserve_ieee_tags(tmp_path, monkeypatch):
    ref = source(tmp_path, [record(1,7), record(2,11,'7ff8000000000002')])
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        import h5py
        original = h5py.File
        def forbidden(*a, **kw):
            raise AssertionError('description opened numerical HDF')
        with monkeypatch.context() as context:
            context.setattr(h5py, 'File', forbidden)
            description = describe_live_history(reader)
            assert description.history['records'] == '2'
            reader.import_history()
        first = read_live_history(reader, maximum_records=1)
        second = read_live_history(reader, cursor=first.cursor)
        assert second.records[0]['core']['fields']['row']['fields']['J']['bits'] == '7ff8000000000002'
        assert second.cursor['row_offset'] == '1' and second.cursor['last_record'] == '2'
        assert second.at_cutoff


def test_foreign_cursor_is_rejected_without_payload_selection(tmp_path):
    ref = source(tmp_path, [record()])
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        cursor = read_live_history(reader).cursor
        cursor['identity']['attempt'] = '2'
        with pytest.raises(ContractError):
            read_live_history(reader, cursor=cursor)


def test_same_open_unlink_safe_rewrite_rejected_and_nofollow(tmp_path):
    from qcl_negf_results._capture_io import copy_closed

    source = tmp_path / "leaf"
    source.write_bytes(b"closed")
    expected = hashlib.sha256(b"closed").hexdigest()
    target = tmp_path / "copy"
    copy_closed(
        source,
        target,
        size=6,
        sha256=expected,
        before_copy=lambda stream: source.unlink(),
    )
    assert target.read_bytes() == b"closed"
    source.write_bytes(b"closed")
    with pytest.raises(ContractError):
        copy_closed(
            source,
            tmp_path / "changed",
            size=6,
            sha256=expected,
            before_copy=lambda stream: source.write_bytes(b"change"),
        )
    linked = tmp_path / "link"
    linked.symlink_to(source)
    with pytest.raises(ContractError):
        copy_closed(linked, tmp_path / "bad", size=6, sha256=expected)
