import unittest

import numpy as np

from code.risk_measure.prepare_bank_prices import extract_code, log_returns
from code.common.quantile_regression import fit_quantile_line, delta_covar


class PipelineTests(unittest.TestCase):
    def test_code_extraction_deduplicates_same_security(self):
        self.assertEqual(extract_code('长江证券(000783):收盘价(后复权):抽取样本'), '000783')
        self.assertIsNone(extract_code('沪深300指数:抽取样本'))

    def test_missing_price_does_not_create_spurious_return(self):
        prices = np.array([10.0, np.nan, 12.0, 12.6])
        returns = log_returns(prices)
        self.assertTrue(np.isnan(returns[1]))
        self.assertTrue(np.isnan(returns[2]))
        self.assertAlmostEqual(returns[3], np.log(12.6 / 12.0))

    def test_quantile_fit_and_delta_use_system_response(self):
        x = np.arange(-5.0, 6.0)
        y = 1.5 + 0.4 * x
        alpha, beta = fit_quantile_line(x, y, 0.95)
        self.assertAlmostEqual(alpha, 1.5, places=6)
        self.assertAlmostEqual(beta, 0.4, places=6)
        self.assertAlmostEqual(delta_covar(x, beta, 0.95), 0.4 * (np.quantile(x, 0.95) - np.median(x)))


if __name__ == '__main__':
    unittest.main()
