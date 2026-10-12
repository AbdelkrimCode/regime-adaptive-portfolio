import pytest
import pandas as pd
from unittest.mock import patch

MOCK_RF_VALUE = 0.0001

# Wide date range covers synthetic test data without depending on CFG.
MOCK_RF_INDEX = pd.bdate_range("1990-01-01", "2050-01-01")

MOCK_RF = pd.Series(MOCK_RF_VALUE, index=MOCK_RF_INDEX)

@pytest.fixture(autouse=True)
def mock_risk_free():
    with patch("backtest.engine.fetch_risk_free", return_value=MOCK_RF):
        yield