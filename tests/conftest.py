import pytest
import pandas as pd
from unittest.mock import patch

MOCK_RF_VALUE = 0.0001

# Wide date range covers synthetic test data without depending on CFG.
MOCK_RF_INDEX = pd.bdate_range("1990-01-01", "2050-01-01")

@pytest.fixture(autouse=True)
def mock_risk_free():
    rf = pd.Series(MOCK_RF_VALUE, index=MOCK_RF_INDEX)
    with patch("optimization.switcher.fetch_risk_free", return_value=rf):
        yield