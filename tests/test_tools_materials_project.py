"""materials_project.py needs a live MP_API_KEY to actually hit the API, unavailable
in this dev environment when this test was first written, so this test mocks MPRester
itself rather than skipping the tool entirely. The field names and search() parameters
this test's fake exercises were independently confirmed real by introspecting the
installed mp-api client's ElectrodeRester/InsertionElectrodeDoc schema, so this test
checks the wrapper's own logic (missing-value handling, id construction), not whether
those field names exist.

Worth noting honestly: this mock did NOT catch a real bug. The tool was first written
calling mpr.materials.electrodes.search(...), which does not exist, the real registered
attribute is mpr.materials.insertion_electrodes. A MagicMock happily accepts any
attribute name with no error, so this test passed the whole time regardless. Only a
real, live call (once a real MP_API_KEY was available) raised the AttributeError that
caught it. A mock proves the wrapper's logic is right; it cannot prove the path into
the real client is right, that needs a live call at least once."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.tools.materials_project import search_electrodes


def _fake_doc(**overrides):
    defaults = dict(
        material_ids=["mp-1001", "mp-1002"], working_ion="Li",
        formula_charge="FePO4", formula_discharge="LiFePO4",
        average_voltage=3.4, capacity_grav=170.0, capacity_vol=590.0, energy_grav=580.0,
        stability_charge=0.0, stability_discharge=0.01, max_delta_volume=0.02,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@patch("agent.tools.materials_project.MPRester")
def test_search_electrodes_maps_real_fields_correctly(mock_mprester_cls, monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "fake-key-for-test")
    mock_mpr = MagicMock()
    mock_mpr.materials.insertion_electrodes.search.return_value = [_fake_doc()]
    mock_mprester_cls.return_value.__enter__.return_value = mock_mpr

    candidates = search_electrodes(working_ion="Li")

    assert len(candidates) == 1
    c = candidates[0]
    assert c.formula_discharge == "LiFePO4"
    assert c.average_voltage == 3.4
    assert c.battery_id == "mp-1001-mp-1002", "id should be built from sorted material_ids"
    print("PASSED: search_electrodes maps real InsertionElectrodeDoc fields correctly")


@patch("agent.tools.materials_project.MPRester")
def test_search_electrodes_handles_missing_material_ids_gracefully(mock_mprester_cls, monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "fake-key-for-test")
    mock_mpr = MagicMock()
    mock_mpr.materials.insertion_electrodes.search.return_value = [_fake_doc(material_ids=None)]
    mock_mprester_cls.return_value.__enter__.return_value = mock_mpr

    candidates = search_electrodes(working_ion="Li")
    assert candidates[0].battery_id == ""
    assert candidates[0].material_ids == []
    print("PASSED: missing material_ids does not crash the mapping")


def test_missing_api_key_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    try:
        search_electrodes(working_ion="Li")
        raise AssertionError("expected a ValueError for a missing MP_API_KEY")
    except ValueError as e:
        assert "MP_API_KEY" in str(e)
        print("PASSED: missing API key raises a clear, actionable error")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
