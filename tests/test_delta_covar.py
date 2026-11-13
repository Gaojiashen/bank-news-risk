"""检查日度估计器是否保持既定CoVaR定义。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from code.risk_measure import compute_daily_delta_covar as daily_delta_covar


class DailyDeltaCoVaRTests(unittest.TestCase):
    def test_daily_dates_and_numpy_output_include_partial_final_month(self):
        with tempfile.TemporaryDirectory(dir=daily_delta_covar.ROOT / "tests") as directory:
            root = Path(directory)
            price_dir = root / "prices"
            output_dir = root / "daily"
            price_dir.mkdir()
            dates = np.array([
                "2022-06-27", "2022-06-28", "2022-06-29", "2022-06-30",
                "2022-07-01", "2022-07-04", "2022-07-05", "2022-07-08",
            ])
            first = np.arange(1.0, 9.0)
            bank_losses = np.column_stack((first, 2 * first))
            system_losses = 1.5 + 0.4 * first
            np.savez_compressed(
                price_dir / "clean_16_bank_prices_returns_and_system.npz",
                dates=dates, bank_names=np.array(["甲行", "乙行"]),
                bank_codes=np.array(["000001", "000002"]),
                bank_log_returns=-bank_losses,
                system_log_return=-system_losses,
            )
            with patch.object(daily_delta_covar, "PRICE_OUT", price_dir), \
                 patch.object(daily_delta_covar, "OUT", output_dir), \
                 patch.object(daily_delta_covar, "WINDOW_DAYS", 5), \
                 patch.object(daily_delta_covar, "MIN_PAIR_DAYS", 4):
                daily_delta_covar.main()

            matrix = np.load(output_dir / "daily_delta_covar.npy")
            with np.load(output_dir / "daily_delta_covar_16_banks.npz") as result:
                np.testing.assert_array_equal(result["dates"], dates)
                np.testing.assert_array_equal(result["delta_covar"], matrix)
                self.assertTrue(np.all(result["status"][:3] == "insufficient"))
                self.assertEqual(result["status"][3, 0], "ok")
                self.assertEqual(result["status"][-1, 0], "ok")
                self.assertAlmostEqual(matrix[3, 0], 0.4 * (np.quantile(first[:4], .95) - np.median(first[:4])))
                self.assertAlmostEqual(matrix[4, 1], 0.2 * (np.quantile(2 * first[:5], .95) - np.median(2 * first[:5])))


if __name__ == "__main__":
    unittest.main()
