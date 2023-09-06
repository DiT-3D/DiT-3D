import numpy as np
import warnings

from scipy.stats import entropy

def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]

def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[np.linalg.norm(grid, axis=1) <= 0.5]

    return grid, spacing

def minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, normalize=True, sess=None, verbose=False, use_sqrt=False,
                             use_EMD=False):
    '''Computes the MMD between two sets of point-clouds.
    Args:
        sample_pcs (numpy array SxKx3): the S point-clouds, each of K points that will be matched and
            compared to a set of "reference" point-clouds.
        ref_pcs (numpy array RxKx3): the R point-clouds, each of K points that constitute the set of
            "reference" point-clouds.
        batch_size (int): specifies how large will the batches be that the compute will use to make
            the comparisons of the sample-vs-ref point-clouds.
        normalize (boolean): if True, the distances are normalized by diving them with
            the number of the points of the point-clouds (n_pc_points).
        use_sqrt: (boolean): When the matching is based on Chamfer (default behavior), if True, the
            Chamfer is computed based on the (not-squared) euclidean distances of the matched point-wise
             euclidean distances.
        sess (tf.Session, default None): if None, it will make a new Session for this.
        use_EMD (boolean: If true, the matchings are based on the EMD.
    Returns:
        A tuple containing the MMD and all the matched distances of which the MMD is their mean.
    '''

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible size of point-clouds.')

    ref_pl, sample_pl, best_in_batch, _, sess = minimum_mathing_distance_tf_graph(n_pc_points, normalize=normalize,
                                                                                  sess=sess, use_sqrt=use_sqrt,
                                                                                  use_EMD=use_EMD)
    matched_dists = []
    for i in range(n_ref):
        best_in_all_batches = []
        if verbose and i % 50 == 0:
            print(i)

        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
            feed_dict = {ref_pl: np.expand_dims(ref_pcs[i], 0), sample_pl: sample_chunk}
            b = sess.run(best_in_batch, feed_dict=feed_dict)
            best_in_all_batches.append(b)
        matched_dists.append(np.min(best_in_all_batches))
    mmd = np.mean(matched_dists)
    return mmd, matched_dists


def coverage(sample_pcs, ref_pcs, batch_size, normalize=True, sess=None, verbose=False, use_sqrt=False, use_EMD=False,
             ret_dist=False):
    '''Computes the Coverage between two sets of point-clouds.
    Args:
        sample_pcs (numpy array SxKx3): the S point-clouds, each of K points that will be matched
            and compared to a set of "reference" point-clouds.
        ref_pcs    (numpy array RxKx3): the R point-clouds, each of K points that constitute the
            set of "reference" point-clouds.
        batch_size (int): specifies how large will the batches be that the compute will use to
            make the comparisons of the sample-vs-ref point-clouds.
        normalize (boolean): if True, the distances are normalized by diving them with
            the number of the points of the point-clouds (n_pc_points).
        use_sqrt  (boolean): When the matching is based on Chamfer (default behavior), if True,
            the Chamfer is computed based on the (not-squared) euclidean distances of the matched
            point-wise euclidean distances.
        sess (tf.Session):  If None, it will make a new Session for this.
        use_EMD (boolean): If true, the matchings are based on the EMD.
        ret_dist (boolean): If true, it will also return the distances between each sample_pcs and
            it's matched ground-truth.
        Returns: the coverage score (int),
                 the indices of the ref_pcs that are matched with each sample_pc
                 and optionally the matched distances of the samples_pcs.
    '''
    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    n_sam, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible Point-Clouds.')

    ref_pl, sample_pl, best_in_batch, loc_of_best, sess = minimum_mathing_distance_tf_graph(n_pc_points,
                                                                                            normalize=normalize,
                                                                                            sess=sess,
                                                                                            use_sqrt=use_sqrt,
                                                                                            use_EMD=use_EMD)
    matched_gt = []
    matched_dist = []
    for i in xrange(n_sam):
        best_in_all_batches = []
        loc_in_all_batches = []

        if verbose and i % 50 == 0:
            print
            i

        for ref_chunk in iterate_in_chunks(ref_pcs, batch_size):
            feed_dict = {ref_pl: np.expand_dims(sample_pcs[i], 0), sample_pl: ref_chunk}
            b, loc = sess.run([best_in_batch, loc_of_best], feed_dict=feed_dict)
            best_in_all_batches.append(b)
            loc_in_all_batches.append(loc)

        best_in_all_batches = np.array(best_in_all_batches)
        b_hit = np.argmin(best_in_all_batches)  # In which batch the minimum occurred.
        matched_dist.append(np.min(best_in_all_batches))
        hit = np.array(loc_in_all_batches)[b_hit]
        matched_gt.append(batch_size * b_hit + hit)

    cov = len(np.unique(matched_gt)) / float(n_ref)

    if ret_dist:
        return cov, matched_gt, matched_dist
    else:
        return cov, matched_gt


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    '''Computes the JSD between two sets of point-clouds, as introduced in the paper ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    '''
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False):
    '''Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    '''
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        p = 0.0
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters

def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)      # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    '''another way of computing JSD'''
    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))