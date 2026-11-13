"""Exact checkpoint checks and bounded gradient replay with CUDA round-off.

Jobs59876/59880 demonstrated exact restored state and forward losses but different
CUDA backward reductions at different gradient scales. Check per-optimizer-group
vector norms/directions and a same-instance repeat control. Check exact restore
and fixed-gradient Adam separately. Report all pointwise deltas as diagnostics.
"""
from copy import deepcopy
import gc
import math

import numpy as np
import torch
from torch import nn

from .engine import (atomic_checkpoint, load_checkpoint, make_model, optimizer_for,
                     rng_state, set_rng, update)
from .protocol import ALPHAS, LOSS_NAMES, alpha_at

RTOL = 1e-5
ATOL = 1e-6
# Engineering one-step reproducibility checks, not convergence guarantees.
# Freeze the same scale-independent limits for every group and both phases.
GRADIENT_RELATIVE_L2_MAX = 0.01
GRADIENT_COSINE_MIN = 0.9999


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)


def compare(left, right):
    """Exact nested comparison and the original1e-5/1e-6 tensor tolerance."""
    differences = []
    tensor_count = 0
    finite = True
    within = True
    max_abs = 0.0

    def walk(a, b, name):
        nonlocal tensor_count, finite, within, max_abs
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            tensor_count += 1
            a, b = a.detach().cpu(), b.detach().cpu()
            if a.shape != b.shape or a.dtype != b.dtype:
                differences.append({'name': name, 'kind': 'tensor_metadata'})
                within = False
                return
            if a.is_floating_point():
                if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                    finite = False
                    within = False
                    differences.append({'name': name, 'kind': 'nonfinite_tensor'})
                    return
                if not torch.equal(a, b):
                    delta = (a.double() - b.double()).abs()
                    peak = delta.max().item() if delta.numel() else 0.0
                    max_abs = max(max_abs, peak)
                    close = bool(torch.allclose(a, b, rtol=RTOL, atol=ATOL))
                    within = within and close
                    differences.append({'name': name, 'kind': 'float_values', 'max_abs': peak,
                        'relative_l2': (delta.norm() / b.double().norm().clamp_min(1e-30)).item(),
                        'different_elements': int((a != b).sum()), 'within_tolerance': close})
            elif not torch.equal(a, b):
                differences.append({'name': name, 'kind': 'nonfloat_values'})
                within = False
        elif isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                differences.append({'name': name, 'kind': 'keys'})
                within = False
            for key in sorted(set(a) & set(b), key=str):
                walk(a[key], b[key], name + '/' + str(key))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                differences.append({'name': name, 'kind': 'sequence'})
                within = False
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, name + '/' + str(i))
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.dtype != b.dtype or not np.array_equal(a, b):
                differences.append({'name': name, 'kind': 'numpy_values'})
                within = False
        elif type(a) is not type(b) or a != b:
            differences.append({'name': name, 'kind': 'scalar', 'left': repr(a), 'right': repr(b)})
            within = False

    walk(left, right, '')
    differences.sort(key=lambda d: d.get('max_abs', 0.0), reverse=True)
    return {'exact': not differences, 'within_tolerance': within, 'finite': finite,
            'rtol': RTOL, 'atol': ATOL, 'max_abs': max_abs,
            'tensor_count': tensor_count, 'mismatch_count': len(differences),
            'outside_tolerance_count': sum(not d.get('within_tolerance', False) for d in differences),
            'largest_differences': differences[:12]}


def runtime(model):
    result = {'cur_epoch': int(model.cur_epoch), 'cur_iter': int(model.cur_iter),
              'pose_step': bool(model.pose_step)}
    for name in ('deform_field', 'txt_generator'):
        module = getattr(model, name)
        result[name] = {'cur_milestone': int(module.cur_milestone), 'act_idx': int(module.act_idx),
                        'activations': [bool(x) for x in module.activations],
                        'code_size': int(module.current_code_size)}
    return result


def gradient_vectors(reference, candidate):
    """Compare common/residual vector norms, directions and exact gradient masks."""
    groups = {name: {'reference_squared': 0.0, 'candidate_squared': 0.0,
                     'delta_squared': 0.0, 'dot': 0.0, 'tensors': 0}
              for name in ('common', 'residual')}
    structural = []
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference or name not in candidate:
            structural.append(name + ': parameter key missing')
            continue
        a, b = reference[name], candidate[name]
        if a is None or b is None:
            if a is not b:
                structural.append(name + ': gradient presence differs')
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            structural.append(name + ': gradient metadata differs')
            continue
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            structural.append(name + ': nonfinite gradient')
            continue
        g = groups['residual' if name.startswith(('vert_fc.', 'uv_fc.')) else 'common']
        a, b = a.detach().cpu().double(), b.detach().cpu().double()
        g['reference_squared'] += a.square().sum().item()
        g['candidate_squared'] += b.square().sum().item()
        g['delta_squared'] += (a - b).square().sum().item()
        g['dot'] += (a * b).sum().item()
        g['tensors'] += 1
    for name, values in list(groups.items()):
        an, bn, dn = [math.sqrt(values[k]) for k in ('reference_squared', 'candidate_squared', 'delta_squared')]
        relative = dn / max(an, bn) if max(an, bn) else 0.0
        cosine = max(-1.0, min(1.0, values['dot'] / (an * bn))) if an and bn else (1.0 if an == bn else 0.0)
        groups[name] = {'reference_l2': an, 'candidate_l2': bn, 'delta_l2': dn,
                        'relative_l2': relative, 'cosine': cosine, 'tensors': values['tensors'],
                        'passed': relative <= GRADIENT_RELATIVE_L2_MAX and cosine >= GRADIENT_COSINE_MIN}
    return {'passed': not structural and all(g['passed'] for g in groups.values()),
            'relative_l2_limit': GRADIENT_RELATIVE_L2_MAX, 'cosine_minimum': GRADIENT_COSINE_MIN,
            'structural_errors': structural[:20], 'groups': groups,
            'pointwise_diagnostic': compare(reference, candidate)}


def runtime_expected(config, epoch_index, cur_iter, pose_step):
    result = {'cur_epoch': int(epoch_index), 'cur_iter': int(cur_iter), 'pose_step': bool(pose_step)}
    for name, key in [('deform_field', 'deform_fields'), ('txt_generator', 'texture_uv')]:
        settings = config['model']['mesh'][key]
        activations = [m <= epoch_index for m in settings['milestones']]
        active = sum(activations)
        result[name] = {'cur_milestone': int(epoch_index), 'act_idx': active,
                        'activations': activations,
                        'code_size': settings['powers'][active - 1] if active else 0}
    return result


def capture_update(model, optimizer, images):
    captured = {'runtime_before': runtime(model)}
    step = optimizer.step

    def observed_step(*args, **kwargs):
        captured['gradients'] = {n: cpu_copy(p.grad) for n, p in model.named_parameters()}
        captured['model_before_step'] = cpu_copy(model.state_dict())
        captured['optimizer_before_step'] = cpu_copy(optimizer.state_dict())
        return step(*args, **kwargs)

    optimizer.step = observed_step
    try:
        captured['row'] = update(model, optimizer, images)
    finally:
        optimizer.step = step
    captured['model'] = cpu_copy(model.state_dict())
    captured['optimizer'] = cpu_copy(optimizer.state_dict())
    captured['neighbors'] = cpu_copy(model.nbr_memory)
    captured['rng'] = cpu_copy(rng_state())
    captured['runtime'] = runtime(model)
    return captured


def replay_adam(model, optimizer, reference):
    """Prove each observed update follows from its exact saved clipped gradient."""
    nn.Module.load_state_dict(model, reference['model_before_step'], strict=True)
    optimizer.load_state_dict(deepcopy(reference['optimizer_before_step']))
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        gradient = reference['gradients'][name]
        parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    optimizer.step()
    return {'model': compare(reference['model'], model.state_dict()),
            'optimizer': compare(reference['optimizer'], optimizer.state_dict())}


def resume_step(path, model, optimizer, scheduler, config, cursor, fingerprint, counters, images, result):
    """Return the resumed live model and populate a report even on failure."""
    result.update({'passed': False, 'checkpoint': str(path),
        'policy': 'Exact saved state; original loss tolerance; per-common/residual gradient vector relativeL2<=1%,cosine>=0.9999 with same-instance control; exact fixed-gradient Adam replay. Pointwise deltas recorded.',
        'tensor_rtol': RTOL, 'tensor_atol': ATOL})
    atomic_checkpoint(path, model, optimizer, scheduler, config, cursor, fingerprint, counters)
    try:
        saved = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location='cpu')
    baseline = capture_update(model, optimizer, images)
    result['phase'] = baseline['row']['phase']
    result['reference_row'] = baseline['row']
    # Use the same instance and input as a control before reconstructing a model.
    # A changed CUDA-backward reduction is observable without a new constructor.
    repeat_cursor, repeat_counters = load_checkpoint(path, model, optimizer, scheduler, config, fingerprint)
    control_restore = {
        'model': compare(saved['model_state'], model.state_dict()),
        'optimizer': compare(saved['optimizer_state'], optimizer.state_dict()),
        'scheduler': compare(saved['scheduler_state'], scheduler.state_dict()),
        'neighbors': compare(saved['neighbor_memory'], model.nbr_memory),
        'rng': compare(saved['rng'], rng_state()),
        'cursor': compare(cursor, repeat_cursor), 'counters': compare(counters, repeat_counters),
        'runtime': compare(runtime_expected(config, saved['cur_epoch'], saved['cur_iter'], saved['pose_step']), runtime(model)),
    }
    result['same_instance_exact_restore'] = control_restore
    if not all(v['exact'] for v in control_restore.values()):
        raise RuntimeError('Exact same-instance checkpoint restoration failed')
    control = capture_update(model, optimizer, images)
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = images.device
    model = make_model(config, device)
    optimizer, scheduler, _ = optimizer_for(model, config)
    loaded_cursor, loaded_counters = load_checkpoint(path, model, optimizer, scheduler, config, fingerprint)
    pre = {
        'model': compare(saved['model_state'], model.state_dict()),
        'optimizer': compare(saved['optimizer_state'], optimizer.state_dict()),
        'scheduler': compare(saved['scheduler_state'], scheduler.state_dict()),
        'neighbors': compare(saved['neighbor_memory'], model.nbr_memory),
        'rng': compare(saved['rng'], rng_state()),
        'cursor': compare(cursor, loaded_cursor), 'counters': compare(counters, loaded_counters),
        'runtime': compare(runtime_expected(config, saved['cur_epoch'], saved['cur_iter'], saved['pose_step']), runtime(model)),
    }
    result['exact_restore'] = pre
    if not all(item['exact'] for item in pre.values()):
        raise RuntimeError('Exact checkpoint restoration failed; inspect resume_check.exact_restore')
    expected_alpha = alpha_at(config, saved['cur_epoch'])
    actual_alpha = [getattr(model, name).item() for name in ALPHAS]
    if not np.allclose(actual_alpha, expected_alpha, rtol=1e-6, atol=1e-9):
        raise RuntimeError('Restored alpha does not match frozen schedule')
    actual = capture_update(model, optimizer, images)
    result['resumed_row'] = actual['row']
    result['loss_checks'] = {}
    for name in LOSS_NAMES:
        a, b = baseline['row']['loss_' + name], actual['row']['loss_' + name]
        tolerance = 1e-5 + 1e-5 * abs(a)
        result['loss_checks'][name] = {'abs_difference': abs(a - b), 'tolerance': tolerance,
                                      'passed': abs(a - b) <= tolerance}
    result['gradient_comparison'] = compare(baseline['gradients'], actual['gradients'])
    result['gradient_vector_check'] = gradient_vectors(baseline['gradients'], actual['gradients'])
    control_checks = {'gradient_vector_check': gradient_vectors(baseline['gradients'], control['gradients']),
        'model_before_step': compare(baseline['model_before_step'], control['model_before_step']),
        'optimizer_before_step': compare(baseline['optimizer_before_step'], control['optimizer_before_step']),
        'runtime': compare(baseline['runtime'], control['runtime']),
        'neighbors': compare(baseline['neighbors'], control['neighbors']),
        'rng': compare(baseline['rng'], control['rng']),
        'loss_checks': {name: abs(baseline['row']['loss_' + name] - control['row']['loss_' + name])
                             <= 1e-5 + 1e-5 * abs(baseline['row']['loss_' + name]) for name in LOSS_NAMES}}
    result['same_instance_control'] = control_checks
    result['state_before_optimizer_step'] = compare(baseline['model_before_step'], actual['model_before_step'])
    result['optimizer_before_step'] = compare(baseline['optimizer_before_step'], actual['optimizer_before_step'])
    result['runtime_after_step'] = compare(baseline['runtime'], actual['runtime'])
    result['neighbors_after_step'] = compare(baseline['neighbors'], actual['neighbors'])
    result['rng_after_step'] = compare(baseline['rng'], actual['rng'])
    # These numerical deltas are evidence, not a replacement checkpoint-load test.
    result['live_weight_comparison'] = compare(baseline['model'], actual['model'])
    result['live_optimizer_comparison'] = compare(baseline['optimizer'], actual['optimizer'])
    checks = [all(v['passed'] for v in result['loss_checks'].values()),
              result['gradient_vector_check']['passed'], control_checks['gradient_vector_check']['passed'],
              all(control_checks['loss_checks'].values()),
              result['live_weight_comparison']['finite'], result['live_optimizer_comparison']['finite']]
    for key in ('state_before_optimizer_step', 'optimizer_before_step', 'runtime_after_step',
                'neighbors_after_step', 'rng_after_step'):
        checks.append(result[key]['exact'])
    for key in ('model_before_step', 'optimizer_before_step', 'runtime', 'neighbors', 'rng'):
        checks.append(control_checks[key]['exact'])
    if not all(checks):
        failures = []
        if not result['gradient_vector_check']['passed']:
            failures.append('reconstructed_gradient_vector')
        if not control_checks['gradient_vector_check']['passed']:
            failures.append('same_instance_gradient_vector')
        for key in ('state_before_optimizer_step', 'optimizer_before_step', 'runtime_after_step', 'neighbors_after_step', 'rng_after_step'):
            if not result[key]['exact']:
                failures.append(key)
        for key in ('model_before_step', 'optimizer_before_step', 'runtime', 'neighbors', 'rng'):
            if not control_checks[key]['exact']:
                failures.append('same_instance_' + key)
        for name, check in result['loss_checks'].items():
            if not check['passed']:
                failures.append('loss_' + name)
        if not all(control_checks['loss_checks'].values()):
            failures.append('same_instance_loss')
        if not result['live_weight_comparison']['finite'] or not result['live_optimizer_comparison']['finite']:
            failures.append('nonfinite_post_update')
        result['failed_checks'] = failures
        raise RuntimeError('Resumed forward/gradient/runtime check failed: ' + ', '.join(failures))
    replay_results = {}
    for label, reference in [('uninterrupted', baseline), ('same_instance_control', control), ('resumed', actual)]:
        replay_results[label] = replay_adam(model, optimizer, reference)
    result['fixed_gradient_adam_replay'] = replay_results
    if not all(v['exact'] for trial in replay_results.values() for v in trial.values()):
        raise RuntimeError('Fixed-gradient Adam replay is not exact; optimizer continuation failed')
    # The final replay put the live model/Adam in the exact resumed post-step state.
    # Restore the rest of that actual post-step state so smoke can continue normally.
    model.sync_epoch(actual['runtime']['cur_epoch'])
    model.cur_iter = actual['runtime']['cur_iter']
    model.pose_step = actual['runtime']['pose_step']
    model.nbr_memory = {k: v.to(device) for k, v in actual['neighbors'].items()}
    set_rng(actual['rng'])
    result['passed'] = True
    result['exact_cuda_trajectory_claimed'] = False
    return model, optimizer, scheduler
