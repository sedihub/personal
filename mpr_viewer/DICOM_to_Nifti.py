""" Reads a DICOM series and writs the image to a NIFTI file. 
"""

import os
import json
import time

import numpy as np
from scipy import stats
import pydicom
import pandas as pd
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


def GetTagAsFloat(_GTds, _GTtagHex1, _GTtagHex2, DefaultValueForException=-1.0):
    try:
        _GTtempStr = float(_GTds[_GTtagHex1,_GTtagHex2].value)
    except:
        _GTtempStr = DefaultValueForException
    finally:
        return _GTtempStr


def GetTagAsStr(_GTds, _GTtagHex1, _GTtagHex2):
    try:
        _GTtempStr = str(_GTds[_GTtagHex1,_GTtagHex2].value)
    except:
        _GTtempStr = ""
    finally:
        return _GTtempStr


def GetTagAsList(_GTds, _GTtagHex1, _GTtagHex2, _length=2):
    try:
        _GTtemp = _GTds[_GTtagHex1,_GTtagHex2].value
    except:
        _GTtemp = []
    finally:
        while len(_GTtemp) != _length:
            _GTtemp.append("")
        return _GTtemp


def GetTagAsJSON(_GTds, _GTtagHex1, _GTtagHex2):    
    """
    Args:
        _GTds (pydicom.dataset): vtkImageData instance.  
        _GTtagHex1 (hex): First part of the tag    
        _GTtagHex2 (hex): Second part of the tag 

    Returns:
        JSON
    """
    try:
        _GTtempJSON = json.loads(str(_GTds[_GTtagHex1,_GTtagHex2].value))
    except:
        _GTtempJSON = json.loads("[]")
    finally:
        return _GTtempJSON


def GetImageMetaData(dicom_series, parse_JSON=False):
    """
    Args:
        dicom_series (dict): A dictionary with slice_z as the key and dataset as value.
        parse_JSON   (bool): Whether or not the JSON block should be parsed. Default is False. 

    Returns:
        a list of metadata
    """
    currentDS = next(iter(dicom_series.values()), None)
    if currentDS is None:
        raise Exception('\"dicom_series\" is empty!')
    tempList=[]

    patientID              = GetTagAsStr(currentDS, 0x0010,0x0020)
    studyInstanceUID       = GetTagAsStr(currentDS, 0x0020,0x000D) 
    seriesInstanceUID      = GetTagAsStr(currentDS, 0x0020,0x000E)
    patientName            = GetTagAsStr(currentDS, 0x0010,0x0010)
    studyID                = GetTagAsStr(currentDS, 0x0020,0x0010)
    sliceThickness         = GetTagAsStr(currentDS, 0x0018,0x0050)
    studyDescription       = GetTagAsStr(currentDS, 0x0008,0x1030)
    seriesDescription      = GetTagAsStr(currentDS, 0x0008,0x103E)
    pixelSpacingXY         = GetTagAsList(currentDS, 0x0028,0x0030)
    windowCentre           = GetTagAsStr(currentDS, 0x0028,0x1050)
    windowWidth            = GetTagAsStr(currentDS, 0x0028,0x1051)
    imageOrientation       = GetTagAsStr(currentDS, 0x0020,0x0037)

    acquisitionDate        = GetTagAsStr(currentDS, 0x0008,0x0022)
    acquisitionDateTime    = GetTagAsStr(currentDS, 0x0008,0x002A)
    institutionName        = GetTagAsStr(currentDS, 0x0008,0x0080)
    institutionCode        = GetTagAsStr(currentDS, 0x0008,0x0082)
    manufacturer           = GetTagAsStr(currentDS, 0x0008,0x0070)
    manufacturersModelName = GetTagAsStr(currentDS, 0x0008,0x1090)
    #sliceLocation          = GetTagAsFloat(currentDS, 0x0020,0x1041)
    #imagePositionPatient   = GetTagAsList(currentDS, 0x0020,0x0032, _length=3)

    tempList = [patientID,
                studyInstanceUID,
                seriesInstanceUID,
                patientName,
                studyID,
                studyDescription,
                seriesDescription,
                len(dicom_series),
                sliceThickness,
                pixelSpacingXY,
                acquisitionDate,
                acquisitionDateTime,
                institutionName,
                institutionCode,
                manufacturer,
                manufacturersModelName,
                windowCentre,
                windowWidth,
                imageOrientation]

    if parse_JSON:
        tempJSON = GetTagAsJSON(currentDS, 0x6819,0x0011)
        tempList.append(json.dumps(tempJSON))

    return tempList


def GetImageVolume(dicom_series, spacing_from_image_position=False):
    """
    Args:
        dicom_series (dict):                A dictionary with slice_z as the key and dataset as value. 
        spacing_from_image_position (bool): Set to true to use the z component of the image positions for z-spacing.

    Returns:
        Tuple ((numpy_array),(float,float,float)): Returns a tuple of a 3D Numpy array containing
        the pixel values and a list of spacing values.
    """

    ## Get dimensions and spacings from the first slice:
    a_slice = next(iter(dicom_series),None)
    if a_slice is None:
        raise Exception('Empty dataset dictionary!')
    tempDS = dicom_series[a_slice]
    dimX = int(tempDS.Rows)
    dimY = int(tempDS.Columns)
    dimZ = len(dicom_series)
    spacing = [float(i) for i in tempDS.PixelSpacing]
    spacing.append(float(tempDS.SliceThickness))

    slope     = float(tempDS.RescaleSlope    )
    intercept = float(tempDS.RescaleIntercept)

    ## Initialize the numpy array
    temp_z_coords=[]
    image_3d_array = np.zeros((dimZ,dimY,dimX),dtype=float)
    for idx,slice_z in enumerate(sorted(dicom_series,reverse=False),start=0):
        temp_imagePositionPatient = GetTagAsList(dicom_series[slice_z], 0x0020,0x0032, _length=3)
        temp_z_coords.append(float(temp_imagePositionPatient[2]))
    
        try:
            tempSliceNumpyArray = dicom_series[slice_z].pixel_array.astype('float32')
        except NotImplementedError as e:
            raise Exception('Compressed DICOM')
        except Exception as e:
            raise Exception('Unexpected Exception while extracting pixel array of slice %s' %(idx+1))
        image_3d_array[idx,:,:] = np.add(np.multiply(tempSliceNumpyArray, slope), intercept)

    if spacing_from_image_position:
        temp_z_spacing_list = [(x - y) for x, y in zip(temp_z_coords,temp_z_coords[1:])]
        temp_z_coords = None
        temp_z_spacing = stats.mode(temp_z_spacing_list)
        temp_z_spacing_list = None
        spacing[2] = float(temp_z_spacing[0]) #np.asscalar(temp_z_spacing[0])

    return image_3d_array, spacing


def ReadDICOMSeries(srcDir, only_read_header=False):    
    """
    Args:
        srcDir (str):            A directory to be searched for dicom images 
        only_read_header (bool): Only read the header!

    Returns:
        A dictionary with slice_z (float) keys and PyDicom.dataset as value.
    """
    dicomFilesDict = {}

    patientID            = None
    studyInstanceUID     = None
    seriesInstanceUID    = None

    for root,dirs,files in os.walk(srcDir):
        for file in files:
            try:
                #print(os.path.join(root,file))
                dataset = pydicom.read_file(os.path.join(root,file), stop_before_pixels=only_read_header)
            except Exception as e:
                print("\tERROR while reading \"", os.path.join(root,file), "\": ", e)      
                print("\tSkipping..."                                                  )
                continue
        
            if patientID is None:
                patientID         = GetTagAsStr(dataset,0x0010,0x0020)
                studyInstanceUID  = GetTagAsStr(dataset,0x0020,0x000D)
                seriesInstanceUID = GetTagAsStr(dataset,0x0020,0x000E)
            else: 
                if patientID != GetTagAsStr(dataset,0x0010,0x0020):
                    raise Exception('Series contains multiple PatientIDs!')
                if studyInstanceUID != GetTagAsStr(dataset,0x0020,0x000D):
                    raise Exception('Series contains multiple StudyInstanceUIDs!')
                if seriesInstanceUID != GetTagAsStr(dataset,0x0020,0x000E):
                    raise Exception('Series contains multiple SeriesInstanceUIDs!')
            
            sliceLocation        = GetTagAsFloat(dataset, 0x0020,0x1041)
            imagePositionPatient = GetTagAsList(dataset, 0x0020,0x0032, _length=3)           
            if imagePositionPatient[2] != "" and imagePositionPatient[2] != sliceLocation:
                sliceZ = float(imagePositionPatient[2])
            else:
                sliceZ = sliceLocation
        
            #print(sliceZ)
            dicomFilesDict[sliceZ] = dataset
    return dicomFilesDict


def WriteNumpyToNifti(np_array, spacing, outputImageFileName):
    # number_of_dimensions = 3
    # pixelType = itk.ctype("float")
    # imageType = itk.Image[ pixelType, number_of_dimensions ]

    np_array = np.flip(np_array,axis=0)
    np_array = np.flip(np_array,axis=1)
    outputImage = itk.GetImageFromArray(np_array.astype(np.float32)) 
    outputImage.SetSpacing(spacing) 
    outputImage.SetOrigin((0.,0.,0.)) 

    # imageCastFilter = itk.CastImageFilter[ outputImage, imageType ].New()
    # imageCastFilter.SetInput(outputImage)
    # 
    imageWriter = itk.ImageFileWriter[outputImage].New()
    imageWriter.SetImageIO(itk.NiftiImageIO.New())
    imageWriter.SetFileName(outputImageFileName)
    imageWriter.SetInput(outputImage)  #imageCastFilter.GetOutput())
    try:
        imageWriter.Update()
    except Exception as e:
        print(BashColours.BOLDRED, "ERROR writing to the ", BashColours.BOLDBLACK, outputImageFileName, BashColours.BOLDRED, " file!", BashColours.RESET)
        print(e)


if __name__ == "__main__":
    # Parse arguments:
    parser = argparse.ArgumentParser(description='DICOM to NIFTI Convertor')
    #
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument("-d", "--dicom", "--DICOM", help="Directory Containing DICOM Images", required=True)
    #
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument("-n", "--nifti", "--NIFTI", help="Output NIFTI filename", default="./output.nii.gz")
    optional_args.add_argument("--dry-run", "--Dry-Run", "--DRY-RUN", 
        help="If provided, skips exporting to nifti.", 
        action="store_true", default=False)
    args = parser.parse_args()

    # Read DICOM files:
    image_series_dict = ReadDICOMSeries(args.dicom, only_read_header=False)

    # Write image to NIFTI:
    if os.path.isfile(args.nifti):
        print("{}WARNING:{} {}{}{} already exists! It will be overwritten.".format(
            BashColours.BOLDRED, BashColours.RESET, 
            BashColours.BOLDBLUE, args.nifti, BashColours.RESET))
    image_series_dict_keys = sorted(image_series_dict.keys())
    image_data_array, spacing = GetImageVolume(image_series_dict)

    if not args.dry_run:
        print("Writing IMAGE to {}{}{}...".format(BashColours.BOLDBLUE, args.nifti, BashColours.RESET))
        tic = time.time()
        WriteNumpyToNifti(image_data_array, spacing, args.nifti)
        toc = time.time()
        print("Done ({}{}{} s.)".format(BashColours.BOLDGREEN, round(toc - tic, 3), BashColours.RESET))
    else:
        print("*** DRY RUN ***")
        print("Slices:")
        print(f"\t{'Z-Location'}\t{'SliceLocation'}\t{'SliceThickness'}")
        for key in sorted(image_series_dict.keys(), reverse=True):
            ds = image_series_dict[key]
            print(f"\t{key}\t{ds.SliceLocation}\t{ds.SliceThickness}")

