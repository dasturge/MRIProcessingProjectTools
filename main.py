#!/usr/bin/env python3
import argparse
import os
import multiprocessing as mp
import subprocess
import sys

from skimage import measure

from funclib import *


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", metavar="IMGPATH", type=str, nargs="+", help="path(s) to T2w to be brain extracted")
    args = parser.parse_args()

    dirname = None
    for image in args.images:
        if not dirname:
            dirname = os.path.dirname(image)
            continue
        if dirname != os.path.dirname(image):
            sys.stderr.write('All images must exist in the same directory. Exiting...')
            sys.exit(1)

    return args


def main():
    """
    functions are found in funclib.py, most image operations
    were written explicitly for this pipeline.  Some are imported
    as writing each algorithm from scratch is time consuming.
    """
    # Script uses ANTs Advanced Normalization Tools for denoising and bias field correction
    # Pycharm is not evaluating my ~/.bashrc if not launched from the terminal, so...
    args = cli()
    t2ws = [os.path.basename(t2w) for t2w in args.images]
    dirname = os.path.dirname(args.images[0])
    os.chdir(dirname)
    # use em if you got em
    if len(t2ws) > 1:
        with mp.Pool(processes=len(t2ws)) as pool:
            pool.map(T2w_skullstrip_pipeline, t2ws)
    else:
        T2w_skullstrip_pipeline(t2ws[0])


def T2w_skullstrip_pipeline(img_name):

    subprocess.check_call("command -v ANTS", shell=True)
    if img_name[-7:] == '.nii.gz':
        img_name = img_name[:-7]

    # skip preprocessing if files already exist
    if os.path.isfile(img_name + '_dn_bf.nii.gz'):
        img_name = img_name + '_dn_bf'
    else:
        # run Rician denoising
        cmd = ['DenoiseImage', '-d', '3', '-i', img_name + '.nii.gz',
               '-v', '-o', img_name + '_dn' + '.nii.gz']
        subprocess.call(" ".join(cmd), shell=True)
        img_name = img_name + '_dn'

        # run Bias Field (inhomogeneity) correction
        cmd = ['N4BiasFieldCorrection', '-d', '3', '-i', img_name + '.nii.gz',
               '-v', '-o', img_name + '_bf' + '.nii.gz']
        subprocess.call(" ".join(cmd), shell=True)
        img_name = img_name + '_bf'

    # start python code
    # load image into python, load array into memory
    img = nib.load(img_name + '.nii.gz')
    arr = img.get_data()

    # initialize structuring element
    struct = np.zeros((3, 3, 3), dtype=np.bool)
    struct[:, 1, 1] = True #      | x |x |   3D structuring element (connected by 6 faces of voxel)
    struct[1, :, 1] = True # | x || x || x |
    struct[1, 1, :] = True #   | x |x |

    # threshold image, get fg binary mask
    thresh = np.ceil(np.percentile(arr[:, :, -10:-1], 99))
    arrmask = np.where(arr < thresh, 0, 1)

    # morphological open to remove background noise on binary mask
    arrmask = erode(arrmask, struct)

    # Apply Otsu's method within roi, take highest intensity class as brain
    otsu = otsun(arr, 2, arrmask)
    rois = np.where(otsu != 3, 0, 1)

    # get largest contiguous region
    labels = measure.label(rois, neighbors=4)  # @todo this is from the skimage package
    roi = largest_roi(labels)

    # morphological close operation on binary mask
    brainmask = morph_close(roi, struct)

    brain = arr * brainmask

    nii_save_like(brain, img, img_name + '_brain' + '.nii.gz')


if __name__ == '__main__':
    # Ideally there would be shell compatible exit statuses, but there's no python guide for this
    main()
    sys.exit(0)

