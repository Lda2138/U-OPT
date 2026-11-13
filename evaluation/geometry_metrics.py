"""Geometry functions copied verbatim from the verified59261 evaluator."""

import hashlib

PROTOCOL = {'version': 'chair_geometry_full_view0_10k_v1', 'scope': 'Numeric geometry definitions retained from59261; applied to seven newly trained final checkpoints.', 'category': '03001627', 'split': 'test', 'selection': 'Every object in the frozen softras_test.lst, in original order.', 'object_limit': None, 'views': [0], 'pred_surface_points': 10000, 'gt_points_max': 10000, 'gt_selection': 'Fixed per-object uniform subset without replacement; shared between methods.', 'seed': 20261021, 'icp': {'method': 'existing gradient_icp', 'iterations': 100, 'lr': 0.01, 'estimate_scale': True, 'anisotropic_scale': True, 'initialization': 'AABB centering and max-extent unit-cube normalization'}, 'camera_alignment': '(vertices @ R_pred + T_pred - T_gt) @ transpose(R_gt), as existing evaluator.', 'gt_coordinate_rotation': [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], 'chamfer': '10 * 0.5 * (mean Euclidean nearest distance in each direction)', 'normal_consistency': 'Symmetric mean absolute cosine at nearest points; row normals use inverse transpose after ICP.', 'fscore_thresholds': [0.01, 0.02], 'fscore_threshold_unit': 'Fraction of full GT pointcloud AABB diagonal; fixed before subset selection.', 'fscore_comparison': '<= threshold; precision pred-to-GT, recall GT-to-pred; outputs in [0,1].', 'mesh_quality_space': 'Raw predicted geometry, centered and divided by its AABB diagonal, before ICP.', 'degenerate_area_threshold': 1e-10, 'triangle_quality': '4*sqrt(3)*area / sum(edge_length**2); 1 is equilateral.', 'self_intersection': 'Not computed; connectivity checks are not intersection tests.', 'aggregation': 'One fixed input per object; report per-object values and arithmetic means separately per method.', 'historical_comparison': 'Same geometry formulas, point counts, ICP and F-score thresholds as59261. New model scores remain a separate table.'}



def unit_normals(normals):
    import numpy as np
    normals = np.asarray(normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3 or not np.isfinite(normals).all():
        raise ValueError('Invalid normal array')
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    if (length <= 1e-12).any():
        raise ValueError('Zero-length normals are not silently scored')
    return normals / length


def nearest(points, reference):
    """Exact Euclidean neighbors; SciPy on cluster, small-array NumPy fallback."""
    import numpy as np
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        if len(points) * len(reference) > 1000000:
            raise RuntimeError('SciPy is required for large pointcloud evaluation')
        dist2 = ((points[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        idx = dist2.argmin(axis=1)
        return np.sqrt(dist2[np.arange(len(points)), idx]), idx
    return cKDTree(reference).query(points, k=1, eps=0)


def surface_metrics(pred, pred_normals, gt, gt_normals, gt_diagonal):
    import numpy as np
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if any(x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all() for x in (pred, gt)):
        raise ValueError('Invalid surface point array')
    if not np.isfinite(gt_diagonal) or gt_diagonal <= 0:
        raise ValueError('Invalid GT diagonal')
    pn, gn = unit_normals(pred_normals), unit_normals(gt_normals)
    if len(pn) != len(pred) or len(gn) != len(gt):
        raise ValueError('Point and normal counts differ')
    d_pg, i_pg = nearest(pred, gt)
    d_gp, i_gp = nearest(gt, pred)
    cos_pg = np.abs(np.sum(pn * gn[i_pg], axis=1)).clip(0, 1)
    cos_gp = np.abs(np.sum(gn * pn[i_gp], axis=1)).clip(0, 1)
    result = {'chamfer_l1_x10': float(5 * (d_pg.mean() + d_gp.mean())),
              'normal_abs_cos': float(0.5 * (cos_pg.mean() + cos_gp.mean()))}
    for frac, tag in ((0.01, '01'), (0.02, '02')):
        precision = float((d_pg <= frac * gt_diagonal).mean())
        recall = float((d_gp <= frac * gt_diagonal).mean())
        result['precision_' + tag] = precision
        result['recall_' + tag] = recall
        result['fscore_' + tag] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return result


def transform_normals(normals, rotation, scale):
    """For row points x' = (x * scale) @ rotation + translation."""
    import numpy as np
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    rotation = np.asarray(rotation, dtype=np.float64)
    if scale.size != 3 or not np.isfinite(scale).all() or (np.abs(scale) < 1e-6).any():
        raise ValueError('Singular or invalid ICP scale')
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError('Invalid ICP rotation')
    linear = np.diag(scale) @ rotation
    # det sign preserves triangle winding under an orientation-reversing map.
    transformed = np.asarray(normals, dtype=np.float64) @ np.linalg.inv(linear).T
    transformed *= np.sign(np.linalg.det(linear))
    return unit_normals(transformed)


def mesh_quality(vertices, faces):
    import numpy as np
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Invalid faces')
    if not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid triangle indices')
    extent = np.ptp(vertices, axis=0)
    diagonal = float(np.linalg.norm(extent))
    if diagonal <= 1e-12:
        raise ValueError('Collapsed mesh')
    v = (vertices - 0.5 * (vertices.max(0) + vertices.min(0))) / diagonal
    tri = v[faces]
    edges = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 1], tri[:, 0]-tri[:, 2]), axis=1)
    area = np.linalg.norm(np.cross(edges[:, 0], -edges[:, 2]), axis=1) * 0.5
    length = np.linalg.norm(edges, axis=2)
    quality = np.divide(4 * np.sqrt(3) * area, (length**2).sum(1),
                        out=np.zeros(len(faces)), where=(length**2).sum(1) > 0)
    angles = []
    for i in range(3):
        a, b = tri[:, (i+1) % 3] - tri[:, i], tri[:, (i+2) % 3] - tri[:, i]
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cos = np.divide((a*b).sum(1), denom, out=np.ones(len(faces)), where=denom > 0)
        angles.append(np.degrees(np.arccos(cos.clip(-1, 1))))
    undirected = np.sort(np.concatenate((faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]])), axis=1)
    _, counts = np.unique(undirected, axis=0, return_counts=True)
    return {'vertices': int(len(vertices)), 'faces': int(len(faces)), 'edges': int(len(counts)),
            'boundary_edges': int((counts == 1).sum()), 'nonmanifold_edges': int((counts > 2).sum()),
            'euler_v_minus_e_plus_f': int(len(vertices)-len(counts)+len(faces)),
            'normalized_min_face_area': float(area.min()),
            'degenerate_face_fraction': float((area <= PROTOCOL['degenerate_area_threshold']).mean()),
            'min_angle_degrees': float(np.min(angles)), 'mean_triangle_quality': float(quality.mean()),
            'min_triangle_quality': float(quality.min()),
            'p01_triangle_quality': float(np.percentile(quality, 1))}


def paired_seed(object_id, view):
    payload = '{}:{}:{}'.format(PROTOCOL['seed'], object_id, view).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)

