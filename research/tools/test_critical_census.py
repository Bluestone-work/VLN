#!/usr/bin/env python3
"""Boundary tests for oracle eligibility, independent from success outcomes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import copy
import unittest
from critical_case_driver import find_interrupts

class InterruptEligibility(unittest.TestCase):
    def record(self):
        return {'high_level_step':1,'scene_id':'scene','action':{'act':4,'ghost_pos':{'__array__':[1.,0.,0.]}},
                'pre_pose':{'position':[0.,0.,0.]},'phases':['backtrack','ghost','ghost','ghost'],
                'primitives':[{'action':1,'collided':True,'pose':{'position':[0.,0.,0.]}} for _ in range(4)]}
    def states(self,reasons=None):
        return [{'episode_id':'1','high_level_step':1,'reasons':reasons or ['collision']}]
    def test_cut_after_executed_ghost_collision(self):
        rec=self.record();before=copy.deepcopy(rec)
        result=find_interrupts({'1':[rec]},self.states(),4)
        self.assertEqual(result[0]['cut_primitive'],2)
        self.assertEqual(rec,before)  # Selecting a cut must not edit the trace.
    def test_last_primitive_collision_is_not_interrupt(self):
        rec=self.record()
        for p in rec['primitives'][:-1]:p['collided']=False
        for i,p in enumerate(rec['primitives']):p['pose']['position']=[float(i+1),0.,0.]
        self.assertEqual(find_interrupts({'1':[rec]},self.states(),4),[])
    def test_backtracking_only_does_not_trigger(self):
        rec=self.record();rec['phases']=['backtrack']*4
        self.assertEqual(find_interrupts({'1':[rec]},self.states(),4),[])
    def test_terminal_or_unflagged_state_not_eligible(self):
        rec=self.record();rec['action']['act']=0
        self.assertEqual(find_interrupts({'1':[rec]},self.states(),4),[])
        rec['action']['act']=4
        self.assertEqual(find_interrupts({'1':[rec]},self.states(['last_nonstop']),4),[])
    def test_stall_requires_forward_moves_not_turns(self):
        rec=self.record();rec['phases']=['ghost']*4
        for p in rec['primitives']:p.update(action=2,collided=False)
        self.assertEqual(find_interrupts({'1':[rec]},self.states(['stall']),4),[])
        for p in rec['primitives']:p['action']=1
        self.assertEqual(find_interrupts({'1':[rec]},self.states(['stall']),4)[0]['cut_primitive'],3)

if __name__=='__main__':unittest.main()
