import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
import trimesh
from pathlib import Path

'''
Custom visualization
'''

def export_to_pc_batch(dir, pcs, colors=None):

    Path(dir).mkdir(parents=True, exist_ok=True)
    for i, xyz in enumerate(pcs):
        if colors is None:
            color = None
        else:
            color = colors[i]
        pcwrite(os.path.join(dir, 'sample_'+str(i)+'.ply'), xyz, color)


def export_to_obj(dir, meshes, transform=lambda v,f:(v,f)):
    '''
    transform: f(vertices, faces) --> transformed (vertices, faces)
    '''
    Path(dir).mkdir(parents=True, exist_ok=True)
    for i, data in enumerate(meshes):
        v, f = transform(data[0], data[1])
        if len(data) > 2:
            v_color = data[2]
        else:
            v_color = None
        mesh = trimesh.Trimesh(v, f, vertex_colors=v_color)
        out = trimesh.exchange.obj.export_obj(mesh)
        with open(os.path.join(dir, 'sample_'+str(i)+'.obj'), 'w') as f:
            f.write(out)
            f.close()

def export_to_obj_single(path, data, transform=lambda v,f:(v,f)):
    '''
    transform: f(vertices, faces) --> transformed (vertices, faces)
    '''
    v, f = transform(data[0], data[1])
    if len(data) > 2:
        v_color = data[2]
    else:
        v_color = None
    mesh = trimesh.Trimesh(v, f, vertex_colors=v_color)
    out = trimesh.exchange.obj.export_obj(mesh)
    with open(path, 'w') as f:
        f.write(out)
        f.close()

def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file.
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2],
            norms[i, 0], norms[i, 1], norms[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2],
        ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyz, rgb=None):
    """Save a point cloud to a polygon .ply file.
    """
    if rgb is None:
        rgb = np.ones_like(xyz) * 128
    rgb = rgb.astype(np.uint8)

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))

'''
Matplotlib Visualization 
'''

def visualize_voxels(out_file, voxels, num_shown=16, threshold=0.5):
    r''' Visualizes voxel data.
    show only first num_shown
    '''
    batch_size =voxels.shape[0]
    voxels = voxels.squeeze(1) > threshold

    num_shown = min(num_shown, batch_size)

    n = int(np.sqrt(num_shown))
    fig = plt.figure(figsize=(20,20))

    for idx, pc in enumerate(voxels[:num_shown]):
        if idx >= n*n:
            break
        pc = voxels[idx]
        ax = fig.add_subplot(n, n, idx + 1, projection='3d')
        ax.voxels(pc, edgecolor='k', facecolors='green', linewidth=0.1, alpha=0.5)
        ax.view_init()
        ax.axis('off')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False, elev=30, azim=225):
    r''' Visualizes point cloud data.
    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1])
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.1, color='k'
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=elev, azim=azim)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualize_pointcloud_batch(path, pointclouds, pred_labels, labels, categories, vis_label=False, target=None,  elev=30, azim=225):
    batch_size = len(pointclouds)
    fig = plt.figure(figsize=(20,20))

    ncols = int(np.sqrt(batch_size))
    nrows = max(1, (batch_size-1) // ncols+1)
    for idx, pc in enumerate(pointclouds):
        if vis_label:
            label = categories[labels[idx].item()]
            pred = categories[pred_labels[idx]]
            colour = 'g' if label == pred else 'r'
        elif target is None:

            colour = 'g'
        else:
            colour = target[idx]
        pc = pc.cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colour, s=5)
        ax.view_init(elev=elev, azim=azim)
        ax.axis('off')
        if vis_label:
            ax.set_title('GT: {0}\nPred: {1}'.format(label, pred))

    plt.savefig(path)
    plt.close(fig)


'''
Plot stats
'''

def plot_stats(output_dir, stats, interval):
    content = stats.keys()
    # f = plt.figure(figsize=(20, len(content) * 5))
    f, axs = plt.subplots(len(content), 1, figsize=(20, len(content) * 5))
    for j, (k, v) in enumerate(stats.items()):
        axs[j].plot(interval, v)
        axs[j].set_ylabel(k)

    f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
    plt.close(f)
