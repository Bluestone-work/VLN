#!/usr/bin/env python3
import unittest
from summarize_interrupt_timing import classify


class ReturnGateTests(unittest.TestCase):
    def row(self, native=0, success=1, ndtw=0., cost=0):
        return {'baseline_metrics': {'success': native}, 'metrics': {'success': success},
                'delta': {'success': success-native, 'ndtw': ndtw, 'steps_taken': cost}}

    def test_success_with_route_degradation_is_not_quality_rescue(self):
        c = classify(self.row(ndtw=-0.01))
        self.assertTrue(c['sr_rescue']); self.assertFalse(c['quality_rescue'])

    def test_cost_gate_is_additional_and_not_hidden(self):
        c = classify(self.row(cost=1))
        self.assertTrue(c['quality_rescue']); self.assertFalse(c['cost_capped_quality_rescue'])

    def test_native_success_never_becomes_rescue(self):
        c = classify(self.row(native=1))
        self.assertFalse(c['sr_rescue']); self.assertTrue(c['identical_metrics'])
        self.assertTrue(classify(self.row(native=1, success=0))['success_loss'])

    def test_metric_tolerance_does_not_tolerate_extra_primitive(self):
        self.assertTrue(classify(self.row(ndtw=-1e-7))['quality_rescue'])
        self.assertFalse(classify(self.row(ndtw=-2e-6))['quality_rescue'])
        self.assertFalse(classify(self.row(cost=1))['cost_capped_quality_rescue'])


if __name__ == '__main__':
    unittest.main()
