import pytest
from Scripts.BuildDatabank.models import BufferCharges
from Scripts.BuildDatabank.build_databank_utils import buffer_charges


def test_buffer_charges_validation():
    """Test that buffer charges are properly validated."""
    # Test valid buffer charges
    BufferCharges(root=buffer_charges)

    # Test invalid pH range (overlapping ranges)
    invalid_charges = buffer_charges.copy()
    invalid_charges["HEPES"]["ph_ranges"] = [
        {"min_ph": 0, "max_ph": 8.0, "charges": [0]},
        {"min_ph": 7.5, "max_ph": 14, "charges": [1]},
    ]
    with pytest.raises(ValueError):
        BufferCharges(root=invalid_charges)

    # Test invalid pH range (not covering full range)
    invalid_charges = buffer_charges.copy()
    invalid_charges["HEPES"]["ph_ranges"] = [
        {"min_ph": 1, "max_ph": 7.0, "charges": [0]},
        {"min_ph": 7.1, "max_ph": 13, "charges": [1]},
    ]
    with pytest.raises(ValueError):
        BufferCharges(root=invalid_charges)

    # Test invalid pH range (max_ph <= min_ph)
    invalid_charges = buffer_charges.copy()
    invalid_charges["HEPES"]["ph_ranges"] = [
        {"min_ph": 0, "max_ph": 7.5, "charges": [0]},
        {"min_ph": 7.5, "max_ph": 14, "charges": [1]},
    ]
    with pytest.raises(ValueError):
        BufferCharges(root=invalid_charges)

    # Test missing ph_ranges for pH-dependent molecule
    invalid_charges = buffer_charges.copy()
    invalid_charges["HEPES"]["ph_ranges"] = None
    with pytest.raises(ValueError):
        BufferCharges(root=invalid_charges)


def test_buffer_charges_structure():
    """Test the structure of buffer charges."""
    model = BufferCharges(root=buffer_charges)

    # Test basic structure
    assert isinstance(model, BufferCharges)
    assert len(model) > 0

    # Test pH-dependent molecule
    hepes = model["HEPES"]
    assert hepes.ph_dependent
    assert isinstance(hepes.ph_ranges, list)

    # Test non-pH-dependent molecule
    water = model["H2O"]
    assert not water.ph_dependent
    assert water.charges == [0]
    assert water.charges_number == [1]
