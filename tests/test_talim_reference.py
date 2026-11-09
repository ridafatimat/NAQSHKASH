"""OCR notation stays tied to the existing repository's Talamat authority."""
import runpy
import pytest
from talim import talim_rules as rules


def test_reference_symbols_and_counts():
    config = runpy.run_path(str(rules.REFERENCE_ROOT / 'config.py'))
    converter = runpy.run_path(str(rules.COUNT_CONVERTER_PATH))
    assert rules.UP_SYMBOLS == config['UP_SYMBOLS']
    assert rules.DOWN_SYMBOLS == config['DOWN_SYMBOLS']
    for count in range(1, 260):
        assert rules.encode_count(count) == converter['getTalamCountString'](count)
    for invalid in (0, 260, True, 1.5):
        with pytest.raises(ValueError):
            rules.validate_count(invalid)
