#!/usr/bin/env python3
import argparse
import os
import multiprocessing as mp
import subprocess
import sys

from skimage import measure

from funclib import *


def cli():
    parser = argparse.ArgumentParser("Runs Otsus Method and Morphological Operations to skull strip neonatal T2w images."
                                     "  Will process subjects in parallel but does not read any additional options"
                                     " in that case.")
    parser.add_argument("images", metavar="IMGPATH", type=str, nargs="+", help="path(s) to T2w to be brain extracted")
    parser.add_argument("--no-preproc", dest="useANTs", action="store_false")
    parser.set_defaults(useANTs=True)
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
    args = cli() # read in command line arguments
    t2ws = [os.path.basename(t2w) for t2w in args.images]
    dirname = os.path.dirname(args.images[0])
    os.chdir(dirname) # outputs will be in same directory as targets

    if len(t2ws) > 1:
        # parellelized if need be
        ncores = len(t2ws) if mp.cpu_count() > len(t2ws) else mp.cpu_count()
        with mp.Pool(processes=ncores) as pool:
            pool.map(T2w_skullstrip_pipeline, t2ws)
    else:
        T2w_skullstrip_pipeline(t2ws[0], useANTs=args.useANTs)


def T2w_skullstrip_pipeline(img_name, **kwargs):

    if img_name[-7:] == '.nii.gz':
        img_name = img_name[:-7]

    useANTs = kwargs.get('useANTs', True)
    if useANTs and os.path.isfile(img_name + '_dn_bf.nii.gz'):
        img_name = img_name + '_dn_bf'
    elif useANTs:
        subprocess.check_call("command -v DenoiseImage", shell=True)
        # run Rician denoising
        cmd = ['DenoiseImage', '-d', '3', '-i', img_name + '.nii.gz',
               '-v', '-o', img_name + '_dn' + '.nii.gz']
        subprocess.call(" ".join(cmd), shell=True)
        img_name = img_name + '_dn'

        subprocess.check_call("command -v N4BiasFieldCorrection", shell=True)
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

    # threshold max radial distance based on histogram
    center = cm(brainmask)
    rad = radial_distance(brainmask, center)
    radhist, edges = np.histogram(rad.flatten(), bins=200)
    idx = np.argmax(radhist[50:]<50) + 50
    brain = np.where(rad < edges[idx], brain, 0)

    nii_save_like(brain, img, img_name + '_brain' + '.nii.gz')


if __name__ == '__main__':
    # Ideally there would be shell compatible exit statuses, but I haven't found the python guide for this
    main()
    sys.exit(0)
