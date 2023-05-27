""" Reads a DICOM series and writs the image to a NIFTI file. 
"""
import os
import json
import time

import numpy as np
import h5py
import itk
import argparse


class BashColours:
    RESET       = "\033[0m"              # Reset
    BLACK       = "\033[30m"             # Black 
    RED         = "\033[31m"             # Red 
    GREEN       = "\033[32m"             # Green 
    BLUE        = "\033[34m"             # Blue 
    BOLDBLACK   = "\033[1m\033[30m"      # Bold Black 
    BOLDRED     = "\033[1m\033[31m"      # Bold Red 
    BOLDGREEN   = "\033[1m\033[32m"      # Bold Green 
    BOLDBLUE    = "\033[1m\033[34m"      # Bold Blue 


def get_h5_image(h5_file, key):
    """Reads h5 file of the image and returns the contents as a numpy file.
    """
    with h5py.File(h5_file, "r") as file:
        if key not in file:
            raise Exception("\"{}\" is not one of the keys ({})!".format(key, file.keys()))
        image_data = file[key][:]

    return image_data


def convert_multimask_bool_to_int(boolian_multimask_array):
    """converts a NumPy array of multimask bool to a mask of
    integers. 
    """
    return boolian_multimask_array.sum(axis=-1)


def mha_to_nifti(mha_src_file, nifti_dest_file, verbose=False):
    """Reads MHA image and writs it as a NIFTI file.
    """
    ## Read MHA image:
    tic = time.time()
    itk_image = itk.imread(mha_src_file)
    toc = time.time()
    if args.verbose:
        print("Read MHA image in {}{}{} s.".format(BashColours.BOLDGREEN, round(toc - tic, 3), BashColours.RESET))
        print("ITK image specs:\n", itk_image)

    ## Write itk image as NIFTI:
    if args.verbose:
        print("Writing IMAGE to {}{}{}...".format(BashColours.BOLDBLUE, nifti_dest_file, BashColours.RESET))
    imageWriter = itk.ImageFileWriter[itk_image].New()
    imageWriter.SetImageIO(itk.NiftiImageIO.New())
    imageWriter.SetFileName(nifti_dest_file)
    imageWriter.SetInput(itk_image)  
    try:
        imageWriter.Update()
    except Exception as e:
        print("{}ERROR writing to the {}{}{}file!{}".format(
            BashColours.BOLDRED, BashColours.BOLDBLACK, nifti_dest_file, BashColours.BOLDRED, BashColours.RESET))
        print(e)
    toc = time.time()
    if args.verbose:
        print("Done! ({}{}{} s)".format(BashColours.BOLDGREEN, round(toc - tic, 3), BashColours.RESET))


def WriteNumpyToNifti(np_array, spacing, output_filename):
    """Writs NumPy array to NIFTI. 
    """
    np_array = np.flip(np_array,axis=0)
    np_array = np.flip(np_array,axis=1)
    outputImage = itk.GetImageFromArray(np_array.astype(np.float32)) 
    outputImage.SetSpacing(spacing) 
    outputImage.SetOrigin((0.0, 0.0, 0.0)) 

    # imageCastFilter = itk.CastImageFilter[ outputImage, imageType ].New()
    # imageCastFilter.SetInput(outputImage)
    # 
    imageWriter = itk.ImageFileWriter[outputImage].New()
    imageWriter.SetImageIO(itk.NiftiImageIO.New())
    imageWriter.SetFileName(output_filename)
    imageWriter.SetInput(outputImage)  #imageCastFilter.GetOutput())
    try:
        imageWriter.Update()
    except Exception as e:
        print("{}ERROR writing to the {}{}{}file!{}".format(
            BashColours.BOLDRED, BashColours.BOLDBLACK, output_filename, BashColours.BOLDRED, BashColours.RESET))
        print(e)


if __name__ == "__main__":
    # Parse arguments:
    parser = argparse.ArgumentParser(description='DICOM to NIFTI Convertor')
    #
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument("-i", "--image", "--IMAGE", help="HDF5 or MHA image", required=True)
    #
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument("-n", "--nifti", "--NIFTI", help="Output NIFTI filename", default="./output.nii.gz")
    optional_args.add_argument("-k", "--h5key", "--H5-KEY", help="The key to use for the H5 file content", default="data")
    optional_args.add_argument("--bool-multimask-to-int", action="store_true", default=False)
    optional_args.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    # Check if IMAGE already exists:
    if not os.path.isfile(args.image):
        raise Exception("{}Input image \"{}\" is not a file!{}".format(
            BashColours.BOLDRED, args.image, BashColours.RESET))

    # Check if NIFTI already exists:
    if os.path.isfile(args.nifti):
        print("{}WARNING:{} {}{}{} already exists! It will be overwritten.".format(
            BashColours.BOLDRED, BashColours.RESET, 
            BashColours.BOLDBLUE, args.nifti, BashColours.RESET))

    input_image_extension = os.path.splitext(args.image)[-1][1:]
    if  input_image_extension == "mha":
        mha_to_nifti(args.image, args.nifti, verbose=args.verbose)
    elif input_image_extension == "h5":
        spacing = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        image_data_array = get_h5_image(args.image, args.h5key)
        if args.verbose:
            print("Loaded array specs: ", image_data_array.shape, image_data_array.dtype)
        if args.bool_multimask_to_int:
            image_data_array = convert_multimask_bool_to_int(image_data_array).astype(np.float64)
            if args.verbose:
                print("Converted array specs: ", image_data_array.shape, image_data_array.dtype)

        if args.verbose:
            print("Writing IMAGE to {}{}{}...".format(BashColours.BOLDBLUE, args.nifti, BashColours.RESET))
        tic = time.time()
        WriteNumpyToNifti(image_data_array, spacing, args.nifti)
        toc = time.time()
        if args.verbose:
            print("Done! ({}{}{} s)".format(BashColours.BOLDGREEN, round(toc - tic, 3), BashColours.RESET))
    else:
        raise Exception(f"Input extension (\"{input_image_extension}\") is neither \"h5\" nor \"mha\"!")
