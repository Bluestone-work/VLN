"""Frozen oracle-label confirmation rule; no dependence on final episode outcomes."""
import math


def confirmation(baseline_window, alternative_window, settings):
    values = [w[k] for w in [baseline_window, alternative_window]
              for k in ['distance_to_goal', 'primitive_count', 'movement_m']]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError('Nonfinite window measurement')
    progress = baseline_window['distance_to_goal'] - alternative_window['distance_to_goal']
    primitives = alternative_window['primitive_count'] - baseline_window['primitive_count']
    movement = alternative_window['movement_m'] - baseline_window['movement_m']
    gates = {'material_progress':progress >= settings['minimum_progress_gain_m'],
             'primitive_cap':primitives <= settings['primitive_delta_max'],
             'path_cap':movement <= settings['movement_delta_max_m']}
    return {'accepted':all(gates.values()),'gates':gates,'extra_goal_progress_m':progress,
            'primitive_delta':primitives,'movement_delta_m':movement}
