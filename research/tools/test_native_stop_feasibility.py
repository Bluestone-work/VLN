import unittest
from audit_native_stop_feasibility import success, failure_flags


class NativeStopFeasibilityTests(unittest.TestCase):
    def test_native_success_boundary_is_inclusive(self):
        self.assertTrue(success(3.0))
        self.assertFalse(success(3.000001))
        for invalid in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):success(invalid)

    def test_interior_arrival_is_not_native_stop_opportunity(self):
        r={'success':False,'oracle_success':True,'boundary_arrival_le_3m':False,'successful_counterfactual_stop_steps':[]}
        f=failure_flags(r)
        self.assertTrue(f['oracle_arrival_without_boundary'])
        self.assertFalse(f['missed_native_stop'])

    def test_native_return_can_help_from_far_state(self):
        r={'success':False,'oracle_success':False,'boundary_arrival_le_3m':False,'successful_counterfactual_stop_steps':[2]}
        self.assertTrue(failure_flags(r)['missed_native_stop'])

    def test_near_pose_is_not_stop_success(self):
        r={'success':False,'oracle_success':True,'boundary_arrival_le_3m':True,'successful_counterfactual_stop_steps':[]}
        self.assertTrue(failure_flags(r)['boundary_arrival_without_native_stop'])
        r['success']=True
        self.assertFalse(any(failure_flags(r).values()))


if __name__=='__main__':unittest.main()
