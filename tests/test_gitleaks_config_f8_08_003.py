"""F8-08-003: .gitleaks.toml must EXTEND the default ruleset, not replace it.

gitleaks v8 uses a supplied config *in place of* its defaults unless
``[extend] useDefault = true`` is present — so generic-api-key,
high-entropy-string and private-key detection never ran. The file allowlist
additionally suppressed every ``*test*.py`` and ``*.md`` file, which is
exactly where scope 17's service creds and the F-08-009 credential docs
live. Four of the eight leaked vendor keys (FMP, MARKETAUX, FRED,
OPENWEATHER) had no matching rule at all.

Source-level (regex on the TOML text — py3.9 has no tomllib), runs under
``pytest --noconftest``.
"""

import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / ".gitleaks.toml"


def _text() -> str:
    return CONFIG.read_text()


class TestF8_08_003_GitleaksConfig:
    def test_extends_default_ruleset(self):
        t = _text()
        assert re.search(r"^\[extend\]", t, re.M), "missing [extend] block"
        assert re.search(r"^\s*useDefault\s*=\s*true", t, re.M), (
            "without useDefault=true the 13 custom rules REPLACE the entire "
            "default ruleset"
        )

    def test_no_blanket_test_file_allowlist(self):
        assert ".*test.*\\.py$" not in _text()
        assert ".*_test\\.py$" not in _text()

    def test_no_blanket_markdown_allowlist(self):
        assert ".*\\.md$" not in _text()

    def test_rules_exist_for_all_eight_leaked_vendor_keys(self):
        t = _text()
        for rule_id in (
            "alpha-vantage-api-key", "finnhub-api-key", "polygon-api-key",
            "news-api-key", "fmp-api-key", "marketaux-api-key",
            "fred-api-key", "openweather-api-key",
        ):
            assert f'id = "{rule_id}"' in t, f"no rule for {rule_id}"
