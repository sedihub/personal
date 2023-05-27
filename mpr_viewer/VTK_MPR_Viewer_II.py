"""A stand alone utility for viewing NIFTI images.
"""

import sys
import os.path
import json

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_numpy_support
import argparse




cursorOff     = False #True


class bashColours:
    RESET       = "\033[0m"              # Reset
    BLACK       = "\033[30m"             # Black 
    RED         = "\033[31m"             # Red 
    GREEN       = "\033[32m"             # Green 
    BLUE        = "\033[34m"             # Blue 
    BOLDBLACK   = "\033[1m\033[30m"      # Bold Black 
    BOLDRED     = "\033[1m\033[31m"      # Bold Red 
    BOLDGREEN   = "\033[1m\033[32m"      # Bold Green  
    BOLDBLUE    = "\033[1m\033[34m"      # Bold Blue 
    RST         = "\x1B[0m"              # RESET


class TextProp:
    def __init__(self,im,actrPosTuple=(0.01,0.975)):
        self.imageViewer = im
        #
        self.sliceTextProp   = None
        self.sliceTextMapper = None
        self.sliceTextActor  = None
        #
        self.AddTextProp(actrPosTuple)

    def AddTextProp(self,actrPosTuple):
        self.sliceTextProp = vtk.vtkTextProperty()
        self.sliceTextProp.SetFontFamilyToCourier()
        self.sliceTextProp.SetFontSize(16)
        self.sliceTextProp.SetColor(0 , 1 , 0)
        self.sliceTextProp.SetVerticalJustificationToBottom()
        self.sliceTextProp.SetJustificationToLeft()
        #
        self.sliceTextMapper = vtk.vtkTextMapper()
        self.sliceTextMapper.SetInput("--/--")
        self.sliceTextMapper.SetTextProperty(self.sliceTextProp)
        #
        self.sliceTextActor = vtk.vtkActor2D()
        self.sliceTextActor.SetMapper(self.sliceTextMapper);
        self.sliceTextActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self.sliceTextActor.GetPositionCoordinate().SetValue(actrPosTuple[0],actrPosTuple[1])
        #
        self.imageViewer.GetRenderer().AddActor2D(self.sliceTextActor)

    def UpdateTextProp(self,msg,render=False):
        self.sliceTextMapper.SetInput(msg)
        if(render):
            self.imageViewer.Render()


class Cursor3D():
    def __init__(self, im):
        self.imageViewer = im
        #
        self.cursor = vtk.vtkCursor3D()
        self.cursor.SetModelBounds(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0)
        self.cursor.AllOff()
        # if(  self.imageViewer.GetSliceOrientation() == vtk.vtkImageViewer2().SLICE_ORIENTATION_YZ):
        #     self.cursor.YShadowsOn()
        #     self.cursor.ZShadowsOn()
        # elif(self.imageViewer.GetSliceOrientation() == vtk.vtkImageViewer2().SLICE_ORIENTATION_XZ):
        #     self.cursor.XShadowsOn()
        #     self.cursor.ZShadowsOn()
        # elif(self.imageViewer.GetSliceOrientation() == vtk.vtkImageViewer2().SLICE_ORIENTATION_XY):
        #     self.cursor.XShadowsOn()
        #     self.cursor.YShadowsOn()
        # else:
        #     print("ERROR: Unrecognized slice orientation!")
        self.cursor.OutlineOff()
        self.cursor.AxesOn()
        self.cursor.TranslationModeOn()
        self.cursor.WrapOff()
        self.cursor.Update()
        #
        self.cursorMapper = vtk.vtkPolyDataMapper()
        self.cursorMapper.SetInputConnection(self.cursor.GetOutputPort())
        #
        self.cursorActor = vtk.vtkActor()
        self.cursorActor.GetProperty().SetColor(0.0, 1.0, 0.0)
        self.cursorActor.ForceOpaqueOff()
        self.cursorActor.SetMapper(self.cursorMapper)
        #
        self.imageViewer.GetRenderer().AddActor(self.cursorActor)
        #
        self.state = True

    def UpdateCursorPosition(self, centre=(0.0, 0.0, 0.0)):
        tolerance = 0.05
        if(  self.imageViewer.GetSliceOrientation() == self.imageViewer.SLICE_ORIENTATION_YZ):
            self.cursor.SetFocalPoint(centre[0]+tolerance,centre[1],centre[2])
        elif(self.imageViewer.GetSliceOrientation() == self.imageViewer.SLICE_ORIENTATION_XZ):
            self.cursor.SetFocalPoint(centre[0],centre[1]+tolerance,centre[2])
        elif(self.imageViewer.GetSliceOrientation() == self.imageViewer.SLICE_ORIENTATION_XY):
            self.cursor.SetFocalPoint(centre[0],centre[1],centre[2]+tolerance)
        else:
            print("ERROR!")    
        self.cursor.Update()

    def UpdateCursorSize(self, size=(1.0, 1.0, 1.0)):
        _bounds = self.cursor.GetModelBounds()
        _centre = (.5*(_bounds[1]+_bounds[0]) ,  .5*(_bounds[3]+_bounds[2]) ,  .5*(_bounds[5]+_bounds[4]))
        self.cursor.SetModelBounds(_centre[0]-_size[0] , centre[0]+_size[0],
                                    _centre[1]-_size[1] , centre[1]+_size[1],
                                    _centre[2]-_size[2] , centre[2]+_size[2])
        self.cursor.Update()

    def CursorVisibility(self):
        if(not self.state):        
            self.cursor.OutlineOff()
            self.cursor.AxesOn()
            self.cursor.TranslationModeOn()
            self.cursor.WrapOff()
            self.cursor.Update()
            #
            self.state = True
        else:
            self.cursor.AllOff()
            self.cursor.Update()
            #
            self.state = False


class CustomInteractorManager:
    def __init__(self, iren):
        iren.RemoveObservers("KeyPressEvent"          )
        iren.RemoveObservers("KeyReleaseEvent"        )
        iren.RemoveObservers("MouseWheelForwardEvent" )
        iren.RemoveObservers("MouseWheelBackwardEvent")
        iren.RemoveObservers("LeftButtonPressEvent"   )
        iren.RemoveObservers("LeftButtonReleaseEvent" )
        #
        iren.AddObserver("KeyPressEvent"            , getattr(self,"KeyPress")          ) 
        iren.AddObserver("KeyReleaseEvent"          , getattr(self,"KeyRelease")        ) 
        iren.AddObserver("LeftButtonPressEvent"     , getattr(self,"LeftButtonPress")   )
        iren.AddObserver("LeftButtonReleaseEvent"   , getattr(self,"LeftButtonRelease") ) 
        iren.AddObserver("MouseMoveEvent"           , getattr(self,"MouseMove")         ) 
        iren.AddObserver("MouseWheelForwardEvent"   , getattr(self,"MouseWheelForward") )
        iren.AddObserver("MouseWheelBackwardEvent"  , getattr(self,"MouseWheelBackward"))
        #
        self.threePlaneView  = None
        self.imageViewer     = None
        self.minSlice        = None
        self.maxSlice        = None
        #
        self.initial_event_position = None
        self._cursor_move_step = 1

    def SetImageViewer(self, mpr, imViewer):
        self.threePlaneView = mpr
        self.imageViewer = imViewer
        #
        self.imageViewer.GetInteractorStyle().RemoveAllObservers()

    def Initialize(self):
        self.minSlice = self.imageViewer.GetSliceMin()
        self.maxSlice = self.imageViewer.GetSliceMax()

    def KeyPress(self, obj, event):
        key = obj.GetKeySym()
        if key == "Up":
            self.threePlaneView.DispatchArrowKeyUpdate(
                self.imageViewer, (0, self._cursor_move_step))
        elif key == "Down":
            self.threePlaneView.DispatchArrowKeyUpdate(
                self.imageViewer, (0, -self._cursor_move_step))
        elif key == "Left":
            self.threePlaneView.DispatchArrowKeyUpdate(
                self.imageViewer, (-self._cursor_move_step, 0))
        elif key == "Right":
            self.threePlaneView.DispatchArrowKeyUpdate(
                self.imageViewer, (self._cursor_move_step, 0))
        elif key == "f" or key == "F":  # Move cursor 10 steps!
            self._cursor_move_step = 10

    def KeyRelease(self, obj, event):
        key = obj.GetKeySym()
        if key == "r" or key == "R":
            self.threePlaneView.DispatchWindowLevelReset(True, True)
        elif key == "w" or key == "W":
            self.threePlaneView.DispatchWindowLevelReset(True, False)
        elif key == "l" or key == "L":
            self.threePlaneView.DispatchWindowLevelReset(False, True)
        elif key == "s" or key == "S":
            self.threePlaneView.TakeScreenshot()
        elif key == "q" or key == "Q":
            self.threePlaneView.Finalize()
            self.threePlaneView.TerminateApp()
        elif key == "c" or key == "C":
            self.threePlaneView.ChangeCurserVisibility()
        elif key == "f" or key == "F":  # Move cursor 10 steps!
            self._cursor_move_step = 1
        elif key == "period":  
            slice = self.imageViewer.GetSlice() + 1
            if slice >= self.minSlice and slice <= self.maxSlice:
                self.threePlaneView.DispatchSliceUpdate(self.imageViewer, slice)
        elif key == "comma":  
            slice = self.imageViewer.GetSlice() - 1
            if slice >= self.minSlice and slice <= self.maxSlice:
                self.threePlaneView.DispatchSliceUpdate(self.imageViewer, slice)

    def LeftButtonPress(self, obj, event):
        if self.initial_event_position is None:
            self.initial_event_position = obj.GetEventPosition()

    def LeftButtonRelease(self, obj, event):
        clickPos = obj.GetEventPosition()
        self.initial_event_position = None
        self.threePlaneView.RefreshCurrentWindowLevel()

    def MouseMove(self, obj, event):
        current_event_position = obj.GetEventPosition()
        # print(f" -> {current_event_position}")
        if(self.initial_event_position is not None):
            self.threePlaneView.DispatchWindowLevelEvent(
                current_event_position,
                self.initial_event_position)
        else:
            self.threePlaneView.DispatchMouseMove(
                self.imageViewer,
                current_event_position)

    def MouseWheelForward(self, obj, event):
        slice = self.imageViewer.GetSlice() + 1
        if slice >= self.minSlice and slice <= self.maxSlice:
            self.threePlaneView.DispatchSliceUpdate(self.imageViewer, slice)

    def MouseWheelBackward(self, obj, event):
        slice = self.imageViewer.GetSlice() - 1
        if slice >= self.minSlice and slice <= self.maxSlice:
            self.threePlaneView.DispatchSliceUpdate(self.imageViewer, slice) 


class ThreePlaneView():
    def __init__(self,parent=None):
        ## Create the three views: Axial Sagittal and Coronal
        self.viewX = vtk.vtkImageViewer2()
        self.viewY = vtk.vtkImageViewer2()
        self.viewZ = vtk.vtkImageViewer2()  
        #
        ## Set their corresponding orientations
        self.viewX.SetSliceOrientationToYZ()
        self.viewY.SetSliceOrientationToXZ()
        self.viewZ.SetSliceOrientationToXY()
        #
        ## Adjust Camera for each viewer:
        self.cameraX = self.viewX.GetRenderer().GetActiveCamera()
        self.cameraX.SetFocalPoint(0, 0, 0)
        self.cameraX.SetPosition(1, 0, 0)
        self.cameraX.SetViewUp(0, 0, -1)
        # self.cameraX.Zoom(1.0)
        #
        self.cameraY = self.viewY.GetRenderer().GetActiveCamera()
        self.cameraY.SetFocalPoint(0, 0, 0)
        self.cameraY.SetPosition(0, 1, 0)
        self.cameraY.SetViewUp(0, 0, -1)
        # self.cameraY.Zoom(1.0)
        #
        self.cameraZ = self.viewZ.GetRenderer().GetActiveCamera()
        self.cameraZ.SetFocalPoint(0, 0, 0)
        self.cameraZ.SetPosition(0, 0, 1)
        self.cameraZ.SetViewUp(0, 1, 0)
        # self.cameraZ.Zoom(1.0)
        ##
        self.renderWindowInteractorX = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractorY = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractorZ = vtk.vtkRenderWindowInteractor()
        #
        self.renderWindowInteractorX.SetRenderWindow(self.viewX.GetRenderWindow())
        self.renderWindowInteractorY.SetRenderWindow(self.viewY.GetRenderWindow())
        self.renderWindowInteractorZ.SetRenderWindow(self.viewZ.GetRenderWindow())
        #
        self.viewX.SetupInteractor(self.renderWindowInteractorX)
        self.viewY.SetupInteractor(self.renderWindowInteractorY)
        self.viewZ.SetupInteractor(self.renderWindowInteractorZ)
        ##
        self.interactorMgrX = CustomInteractorManager(self.renderWindowInteractorX)
        self.interactorMgrY = CustomInteractorManager(self.renderWindowInteractorY)
        self.interactorMgrZ = CustomInteractorManager(self.renderWindowInteractorZ)
        #
        self.interactorMgrX.SetImageViewer(self, self.viewX)
        self.interactorMgrY.SetImageViewer(self, self.viewY)
        self.interactorMgrZ.SetImageViewer(self, self.viewZ)
        ##
        self.textPropX = TextProp(self.viewX)
        self.textPropY = TextProp(self.viewY)
        self.textPropZ = TextProp(self.viewZ)
        ##
        if cursorOff:
            self.cursorX = None
            self.cursorY = None
            self.cursorZ = None
        else:
            self.cursorX = Cursor3D(self.viewX)
            self.cursorY = Cursor3D(self.viewY)
            self.cursorZ = Cursor3D(self.viewZ)
        ##
        self.maskActorX = None
        self.maskActorY = None
        self.maskActorZ = None
        #
        self.pickerX = vtk.vtkPropPicker()
        self.pickerY = vtk.vtkPropPicker()
        self.pickerZ = vtk.vtkPropPicker()
        #
        self.imageDataSpacing = None
        self.imageDataDimensions = None
        #
        self.lastImageCoordinates = [0, 0, 0]
        #
        self.position = [0.0, 0.0, 0.0]
        #
        self.initial_window_width = None
        self.initial_window_level = None
        #
        self.current_window_width = None
        self.current_window_level = None

    def SetImageDate(self,image_data):
        self.viewX.SetInputData(image_data)
        self.viewY.SetInputData(image_data)
        self.viewZ.SetInputData(image_data)
        #
        self.interactorMgrX.Initialize()
        self.interactorMgrY.Initialize()
        self.interactorMgrZ.Initialize()
        #
        self.imageDataSpacing    = image_data.GetSpacing()
        self.imageDataDimensions = image_data.GetDimensions()

    def SetViewersWindowName(self,window_title=" ¯\\_(ツ)_/¯"):
        self.viewX.GetRenderWindow().SetWindowName(" ".join([window_title,"(X)"]))
        self.viewY.GetRenderWindow().SetWindowName(" ".join([window_title,"(Y)"]))
        self.viewZ.GetRenderWindow().SetWindowName(" ".join([window_title,"(Z)"]))

    def SetViewersWindowSize(self, x_pixels=1024, y_pixels=1024):
        self.renderWindowInteractorX.GetRenderWindow().SetSize(x_pixels, y_pixels)
        self.renderWindowInteractorY.GetRenderWindow().SetSize(x_pixels, y_pixels)
        self.renderWindowInteractorZ.GetRenderWindow().SetSize(x_pixels, y_pixels)

    def SetViewersBackgroundColour(self, r=0.125, g=0.0, b=0.125):
        self.viewX.GetRenderer().SetBackground(r, g, b)
        self.viewY.GetRenderer().SetBackground(r, g, b)
        self.viewZ.GetRenderer().SetBackground(r, g, b)

    def SetViewersWindowLevel(self, level, width):
        if (width is not None) and (level is not None):
            self.initial_window_width = width
            self.initial_window_level = level
            #
            self.viewX.SetColorLevel(level)
            self.viewY.SetColorLevel(level)
            self.viewZ.SetColorLevel(level)
            #
            self.viewX.SetColorWindow(width)
            self.viewY.SetColorWindow(width)
            self.viewZ.SetColorWindow(width)
        else:
            self.initial_window_width = self.viewX.GetColorWindow()
            self.initial_window_level = self.viewX.GetColorLevel()
        #
        self.current_window_width = self.initial_window_width
        self.current_window_level = self.initial_window_level

    def SetInterpolation(self,interpolationType="Nearest"):
        if interpolationType == "Cubic":
            self.viewX.GetImageActor().GetProperty().SetInterpolationTypeToCubic()
            self.viewY.GetImageActor().GetProperty().SetInterpolationTypeToCubic()
            self.viewZ.GetImageActor().GetProperty().SetInterpolationTypeToCubic()
        elif interpolationType == "Linear":
            self.viewX.GetImageActor().GetProperty().SetInterpolationTypeToLinear()
            self.viewY.GetImageActor().GetProperty().SetInterpolationTypeToLinear()
            self.viewZ.GetImageActor().GetProperty().SetInterpolationTypeToLinear()
        else:
            self.viewX.GetImageActor().GetProperty().SetInterpolationTypeToNearest()
            self.viewY.GetImageActor().GetProperty().SetInterpolationTypeToNearest()
            self.viewZ.GetImageActor().GetProperty().SetInterpolationTypeToNearest()

    def SetMaskData(self, mask_Data, colours_List):
        maskLookUpTable = vtk.vtkLookupTable()
        maskLookUpTable.SetNumberOfTableValues(len(colours_List) + 1)
        maskLookUpTable.SetRange(0.0, float(1 + len(colours_List)))
        maskLookUpTable.SetTableValue(0 , 0.0, 0.0, 0.0, 0.0);
        if len(colours_List[0]) == 4:
            for idx , c in enumerate(colours_List):
                maskLookUpTable.SetTableValue(idx + 1, c[0], c[1], c[2], c[3])
        elif len(colours_List[0]) == 5:
            for c in colours_List:
                maskLookUpTable.SetTableValue(c[0], c[1], c[2], c[3], c[4])
        else:
            return
        ##
        maskMapperX = vtk.vtkImageMapToColors()
        maskMapperX.SetLookupTable(maskLookUpTable);
        maskMapperX.PassAlphaToOutputOn();
        maskMapperX.SetInputData(mask_Data);
        #
        maskMapperY = vtk.vtkImageMapToColors()
        maskMapperY.SetLookupTable(maskLookUpTable);
        maskMapperY.PassAlphaToOutputOn();
        maskMapperY.SetInputData(mask_Data);
        #
        maskMapperZ = vtk.vtkImageMapToColors()
        maskMapperZ.SetLookupTable(maskLookUpTable);
        maskMapperZ.PassAlphaToOutputOn();
        maskMapperZ.SetInputData(mask_Data);
        ##
        self.maskActorX = vtk.vtkImageActor()
        self.maskActorX.GetMapper().SetInputConnection(maskMapperX.GetOutputPort())
        self.maskActorX.GetMapper().SliceFacesCameraOn()
        self.maskActorX.GetMapper().SliceAtFocalPointOn()
        self.maskActorX.InterpolateOff()
        self.maskActorX.Update()
        #
        self.maskActorY = vtk.vtkImageActor()
        self.maskActorY.GetMapper().SetInputConnection(maskMapperY.GetOutputPort())
        self.maskActorY.GetMapper().SliceFacesCameraOn()
        self.maskActorY.GetMapper().SliceAtFocalPointOn()
        self.maskActorY.InterpolateOff()
        self.maskActorY.Update()
        #
        self.maskActorZ = vtk.vtkImageActor()
        self.maskActorZ.GetMapper().SetInputConnection(maskMapperZ.GetOutputPort())
        self.maskActorZ.GetMapper().SliceFacesCameraOn()
        self.maskActorZ.GetMapper().SliceAtFocalPointOn()
        self.maskActorZ.InterpolateOff()
        self.maskActorZ.Update()
        ##
        self.viewX.GetRenderer().AddActor(self.maskActorX)
        self.viewY.GetRenderer().AddActor(self.maskActorY)
        self.viewZ.GetRenderer().AddActor(self.maskActorZ)
        #
        self.Render()

    def UpdateMasks(self,currentImageViewer=None):
        if self.maskActorX is not None and \
           self.maskActorY is not None and \
           self.maskActorZ is not None :
            if currentImageViewer is not None:
                self.maskActorX.SetDisplayExtent(self.viewX.GetImageActor().GetDisplayExtent())
                self.maskActorY.SetDisplayExtent(self.viewY.GetImageActor().GetDisplayExtent())
                self.maskActorZ.SetDisplayExtent(self.viewZ.GetImageActor().GetDisplayExtent())
            self.maskActorX.Update()
            self.maskActorY.Update()
            self.maskActorZ.Update()

    def DispatchMouseMove(self, imViewer, eventPos):
        if imViewer == self.viewX:
            self.pickerX.Pick(eventPos[0], eventPos[1], 0, self.viewX.GetRenderer())
            if self.pickerX.GetPath():
                position = self.pickerX.GetPickPosition()
                image_coordinate = (self.viewX.GetSlice() - self.viewX.GetSliceMin(),
                                    round(position[1] / self.imageDataSpacing[1]),
                                    round(position[2] / self.imageDataSpacing[2]))
                self.position = [
                    self.viewX.GetSlice() * self.imageDataSpacing[0], position[1], position[2] ]
                self.viewY.SetSlice(image_coordinate[1])
                self.viewZ.SetSlice(image_coordinate[2])
            else:
                image_coordinate = (-1 , -1 , -1)
            # print(f"---> {image_coordinate}")
        elif imViewer == self.viewY:
            self.pickerY.Pick(eventPos[0], eventPos[1], 0, self.viewY.GetRenderer())
            if self.pickerY.GetPath():
                position = self.pickerY.GetPickPosition()
                image_coordinate = (round(position[0] / self.imageDataSpacing[0]),
                                    self.viewY.GetSlice() - self.viewY.GetSliceMin(),
                                    round(position[2] / self.imageDataSpacing[2]))
                self.position = [
                    position[0] , self.viewY.GetSlice() * self.imageDataSpacing[1], position[2]]
                self.viewX.SetSlice(image_coordinate[0])
                self.viewZ.SetSlice(image_coordinate[2])
            else:
                image_coordinate = (-1 , -1 , -1)
            # print(f"---> {image_coordinate}")
        elif imViewer == self.viewZ:
            self.pickerZ.Pick(eventPos[0], eventPos[1], 0, self.viewZ.GetRenderer())
            if self.pickerZ.GetPath():
                position = self.pickerZ.GetPickPosition()
                image_coordinate = (round(position[0] / self.imageDataSpacing[0]),
                                    round(position[1] / self.imageDataSpacing[1]),
                                    self.viewZ.GetSlice() - self.viewZ.GetSliceMin())
                self.position = [
                    position[0], position[1], self.viewZ.GetSlice() * self.imageDataSpacing[2]]
                self.viewX.SetSlice(image_coordinate[0])
                self.viewY.SetSlice(image_coordinate[1])
            else:
                image_coordinate = (-1 , -1 , -1)
            # print(f"---> {image_coordinate}")
        else:
            print("ERROR: Unspecified vtkImageViewer!")
            return
        # 
        if image_coordinate[0] >= 0 and image_coordinate[0] < self.imageDataDimensions[0] and\
           image_coordinate[1] >= 0 and image_coordinate[1] < self.imageDataDimensions[1] and\
           image_coordinate[2] >= 0 and image_coordinate[2] < self.imageDataDimensions[2]:
            voxVal = self.viewX.GetInput().GetScalarComponentAsDouble(
                image_coordinate[0] , image_coordinate[1] , image_coordinate[2] , 0)
            self.textPropX.UpdateTextProp("".join(
                [ "(" , str(image_coordinate[0]+1) , "/" , str(self.imageDataDimensions[0]) , " , " ,\
                        str(image_coordinate[1]+1) , "/" , str(self.imageDataDimensions[1]) , " , " ,\
                        str(image_coordinate[2]+1) , "/" , str(self.imageDataDimensions[2]) , ")    " , str(voxVal) ]))
            self.textPropY.UpdateTextProp("".join(
                [ "(" , str(image_coordinate[0]+1) , "/" , str(self.imageDataDimensions[0]) , " , " ,\
                        str(image_coordinate[1]+1) , "/" , str(self.imageDataDimensions[1]) , " , " ,\
                        str(image_coordinate[2]+1) , "/" , str(self.imageDataDimensions[2]) , ")    " , str(voxVal) ]))
            self.textPropZ.UpdateTextProp("".join(
                [ "(" , str(image_coordinate[0]+1) , "/" , str(self.imageDataDimensions[0]) , " , " ,\
                        str(image_coordinate[1]+1) , "/" , str(self.imageDataDimensions[1]) , " , " ,\
                        str(image_coordinate[2]+1) , "/" , str(self.imageDataDimensions[2]) , ")    " , str(voxVal) ]))
            #
            self.lastImageCoordinates = list(image_coordinate)
            #
            if(not cursorOff):
                self.cursorX.UpdateCursorPosition(self.position)
                self.cursorY.UpdateCursorPosition(self.position)
                self.cursorZ.UpdateCursorPosition(self.position)
        else:
            self.textPropX.UpdateTextProp("--")
            self.textPropY.UpdateTextProp("--")
            self.textPropZ.UpdateTextProp("--")
        #
        self.UpdateMasks(imViewer)
        self.Render()

    def DispatchArrowKeyUpdate(self, imViewer, move=(0, 0)):
        if imViewer == self.viewX:
            self.viewY.SetSlice(self.viewY.GetSlice() - move[0])
            self.viewZ.SetSlice(self.viewZ.GetSlice() - move[1])
            self.lastImageCoordinates[1] = self.viewY.GetSlice() - self.viewY.GetSliceMin()
            self.lastImageCoordinates[2] = self.viewZ.GetSlice() - self.viewZ.GetSliceMin()
        elif imViewer == self.viewY:
            self.viewX.SetSlice(self.viewX.GetSlice() + move[0])
            self.viewZ.SetSlice(self.viewZ.GetSlice() - move[1])
            self.lastImageCoordinates[0] = self.viewX.GetSlice() - self.viewX.GetSliceMin()
            self.lastImageCoordinates[2] = self.viewZ.GetSlice() - self.viewZ.GetSliceMin()
        elif imViewer == self.viewZ:
            self.viewX.SetSlice(self.viewX.GetSlice() + move[0])
            self.viewY.SetSlice(self.viewY.GetSlice() + move[1])
            self.lastImageCoordinates[0] = self.viewX.GetSlice() - self.viewX.GetSliceMin()
            self.lastImageCoordinates[1] = self.viewY.GetSlice() - self.viewY.GetSliceMin()
        else:
            print("ERROR: Unspecified vtkImageViewer!")
            return

        self.position[0] = self.viewX.GetSlice() * self.imageDataSpacing[0]
        self.position[1] = self.viewY.GetSlice() * self.imageDataSpacing[1]
        self.position[2] = self.viewZ.GetSlice() * self.imageDataSpacing[2]
        #
        voxVal = self.viewX.GetInput().GetScalarComponentAsDouble(
            self.lastImageCoordinates[0], 
            self.lastImageCoordinates[1], 
            self.lastImageCoordinates[2], 
            0)
        self.textPropX.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self.imageDataDimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self.imageDataDimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self.imageDataDimensions[2]) , ")    " , str(voxVal) ]))
        self.textPropY.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self.imageDataDimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self.imageDataDimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self.imageDataDimensions[2]) , ")    " , str(voxVal) ]))
        self.textPropZ.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self.imageDataDimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self.imageDataDimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self.imageDataDimensions[2]) , ")    " , str(voxVal) ]))
        #
        if not cursorOff:
            self.cursorX.UpdateCursorPosition(self.position)
            self.cursorY.UpdateCursorPosition(self.position)
            self.cursorZ.UpdateCursorPosition(self.position)
        #
        self.UpdateMasks(imViewer)
        self.Render()    

    def DispatchSliceUpdate(self, imViewer, slice):
        if imViewer == self.viewX:
            self.viewX.SetSlice(slice)
            self.lastImageCoordinates[0] = slice - self.viewX.GetSliceMin()
            self.position[0] = self.viewX.GetSlice() * self.imageDataSpacing[0]
        elif imViewer == self.viewY:
            self.viewY.SetSlice(slice)
            self.lastImageCoordinates[1] = slice - self.viewY.GetSliceMin()
            self.position[1] = self.viewY.GetSlice() * self.imageDataSpacing[1]
        elif imViewer == self.viewZ:
            self.viewZ.SetSlice(slice)
            self.lastImageCoordinates[2] = slice - self.viewZ.GetSliceMin()
            self.position[2] = self.viewZ.GetSlice() * self.imageDataSpacing[2]
        else:
            print("ERROR: Unspecified vtkImageViewer!")
            return
        #
        voxVal = self.viewX.GetInput().GetScalarComponentAsDouble(
            self.lastImageCoordinates[0] , self.lastImageCoordinates[1] , self.lastImageCoordinates[2] , 0)
        self.textPropX.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self.imageDataDimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self.imageDataDimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self.imageDataDimensions[2]) , ")    " , str(voxVal) ]))
        self.textPropY.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self.imageDataDimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self.imageDataDimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self.imageDataDimensions[2]) , ")    " , str(voxVal) ]))
        self.textPropZ.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self.imageDataDimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self.imageDataDimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self.imageDataDimensions[2]) , ")    " , str(voxVal) ]))
        #
        if not cursorOff:
            self.cursorX.UpdateCursorPosition(self.position)
            self.cursorY.UpdateCursorPosition(self.position)
            self.cursorZ.UpdateCursorPosition(self.position)
        #
        self.UpdateMasks(imViewer)
        self.Render()

    def RefreshCurrentWindowLevel(self):
        self.current_window_level = self.viewX.GetColorLevel()
        self.current_window_width = self.viewX.GetColorWindow()

    def DispatchWindowLevelEvent(self,current_dispatched_event_position,initial_dispatched_event_positions):
        level = self.current_window_level + round(
            (current_dispatched_event_position[0] - initial_dispatched_event_positions[0]) / 0.25)
        width = self.current_window_width + round(
            (current_dispatched_event_position[1] - initial_dispatched_event_positions[1]) / 0.25)
        # width = max(self.current_window_width + round(
        #    (current_dispatched_event_position[1]-initial_dispatched_event_positions[1])/.25) , 10.)
        #
        self.viewX.SetColorLevel(level)
        self.viewY.SetColorLevel(level)
        self.viewZ.SetColorLevel(level)
        #
        self.viewX.SetColorWindow(width)
        self.viewY.SetColorWindow(width)
        self.viewZ.SetColorWindow(width)
        #
        self.Render()

    def DispatchWindowLevelReset(self,reset_window,reset_level):
        if reset_window:
            self.viewX.SetColorWindow(self.initial_window_width)
            self.viewY.SetColorWindow(self.initial_window_width)
            self.viewZ.SetColorWindow(self.initial_window_width)
        #
        if reset_level :
            self.viewX.SetColorLevel(self.initial_window_level)
            self.viewY.SetColorLevel(self.initial_window_level)
            self.viewZ.SetColorLevel(self.initial_window_level)
        #
        self.RefreshCurrentWindowLevel()
        self.Render()

    def TakeScreenshot(self):
        ## Screenshot PNGImageWriter and WindowToImageFilter:
        screenshotImageWRiterX = vtk.vtkPNGWriter()
        screenshotImageWRiterY = vtk.vtkPNGWriter()
        screenshotImageWRiterZ = vtk.vtkPNGWriter()
        #
        screenshotX = vtk.vtkWindowToImageFilter()
        screenshotY = vtk.vtkWindowToImageFilter()
        screenshotZ = vtk.vtkWindowToImageFilter()
        #
        screenshotX.SetInput(self.viewX.GetRenderWindow())
        screenshotY.SetInput(self.viewY.GetRenderWindow())
        screenshotZ.SetInput(self.viewZ.GetRenderWindow())
        #
        screenshotX.SetScale(3, 3)
        screenshotY.SetScale(3, 3)
        screenshotZ.SetScale(3, 3)
        #
        screenshotX.SetInputBufferTypeToRGBA()
        screenshotY.SetInputBufferTypeToRGBA()
        screenshotZ.SetInputBufferTypeToRGBA()
        #
        # screenshotX.ReadFrontBufferOff() #By default is On
        # screenshotY.ReadFrontBufferOff() #By default is On
        # screenshotZ.ReadFrontBufferOff() #By default is On
        #
        screenshotX.Update()
        screenshotY.Update()
        screenshotZ.Update()
        #
        temp_filename_x = "".join(
            [ "./" , str(self.viewX.GetRenderWindow().GetWindowName()).split(" (X)")[0] , 
              "__X-" , str(self.viewX.GetSlice()) , ".png" ])
        temp_filename_y = "".join(
            [ "./" , str(self.viewY.GetRenderWindow().GetWindowName()).split(" (Y)")[0] , 
              "__Y-" , str(self.viewY.GetSlice()) , ".png" ])
        temp_filename_z = "".join(
            [ "./" , str(self.viewZ.GetRenderWindow().GetWindowName()).split(" (Z)")[0] , 
              "__Z-" , str(self.viewZ.GetSlice()) , ".png" ])
        #
        screenshotImageWRiterX.SetFileName(temp_filename_x)
        screenshotImageWRiterY.SetFileName(temp_filename_y)
        screenshotImageWRiterZ.SetFileName(temp_filename_z)
        #
        screenshotImageWRiterX.SetInputConnection(screenshotX.GetOutputPort())
        screenshotImageWRiterY.SetInputConnection(screenshotY.GetOutputPort())
        screenshotImageWRiterZ.SetInputConnection(screenshotZ.GetOutputPort())
        #
        screenshotImageWRiterX.Write()
        screenshotImageWRiterY.Write()
        screenshotImageWRiterZ.Write()

    def ChangeCurserVisibility(self):
        if not cursorOff:
            self.cursorX.CursorVisibility()
            self.cursorY.CursorVisibility()
            self.cursorZ.CursorVisibility()
        self.Render()


    def Initialize(self):
        self.renderWindowInteractorX.Initialize()
        self.renderWindowInteractorY.Initialize()
        self.renderWindowInteractorZ.Initialize()

    def Render(self):
        self.viewX.Render()
        self.viewY.Render()
        self.viewZ.Render()

    def Start(self):
        self.renderWindowInteractorX.Start()
        self.renderWindowInteractorY.Start()
        self.renderWindowInteractorZ.Start()
        
    def Finalize(self):
        self.renderWindowInteractorX.GetRenderWindow().Finalize()
        self.renderWindowInteractorY.GetRenderWindow().Finalize()
        self.renderWindowInteractorZ.GetRenderWindow().Finalize()

    def TerminateApp(self):
        self.renderWindowInteractorX.TerminateApp()
        self.renderWindowInteractorY.TerminateApp()
        self.renderWindowInteractorZ.TerminateApp()
        #
        quit()


def main():
    # Parse arguments:
    parser = argparse.ArgumentParser(description="VTK MPR Viewer")
    #
    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument("-i", "--image", "--IMAGE", help="NIFTI, VTI or NNRD Image", required=True)
    #
    optional_args = parser.add_argument_group("Optional Arguments")
    optional_args.add_argument("-m", "--mask", "--MASK", help="NIFTI, VTI or NNRD Mask", default="")
    optional_args.add_argument("-c", "--color-map", "--COLOR-MAP", help="Color map for mask", default=[])
    optional_args.add_argument("--window-level", "--wl", "--WL", help="Window center", default=None, type=float)
    optional_args.add_argument("--window-width", "--ww", "--WW", help="Window width", default=None, type=float)
    optional_args.add_argument("--window-size", "--WS", help="Window size", default=[], type=list)
    optional_args.add_argument("-t", "--window-title", "--WINDOW-TITLE", help="Window Title", default="MPR Viewer")
    optional_args.add_argument("-b", "--background", "--BACKGROUND", 
        help="Background RGB color", default=[0.0, 0.0, 0.25], type=list)
    optional_args.add_argument("--interpolation", 
        help="Interpolation (\"Nearest\", \"Linear\", or \"Cubic\")", default="Nearest")
    args = parser.parse_args()

    # Print arguments:
    print("Arguments:\n", json.dumps(vars(args), indent=4), end="\n\n\n")


    # Load image data:
    imageData = None
    if args.image.split(".")[-1] == "vti":
        if not os.path.isfile(args.image):
            print("{}ERROR: \"{}{}{}\" does not exist!{}".format(
                bashColours.BOLDRED, bashColours.BOLDBLACK, args.image, bashColours.BOLDRED, bashColours.RESET))
            quit()
        imageReader = vtk.vtkXMLImageDataReader()
        imageReader.SetFileName(args.image)
        try:
            imageReader.Update()
            imageData = imageReader.GetOutput()
        except:
            print(bashColours.BOLDRED, "ERROR reading the ", bashColours.BOLDBLACK, args.image, bashColours.BOLDRED, " file!\nAborting...", bashColours.RESET)
            quit()
    elif args.image.split(".")[-1] == "nrrd":
        if not os.path.isfile(args.image):
            print("{}ERROR: \"{}{}{}\" does not exist!{}".format(
                bashColours.BOLDRED, bashColours.BOLDBLACK, args.image, bashColours.BOLDRED, bashColours.RESET))
            quit()
        imageReader = vtk.vtkNrrdReader()
        imageReader.SetFileName(args.image)
        try:
            imageReader.Update()
            imageData = imageReader.GetOutput()
        except:
            print(bashColours.BOLDRED, "ERROR reading the ", bashColours.BOLDBLACK, args.image, bashColours.BOLDRED, " file!\nAborting...", bashColours.RESET)
            quit()
    elif args.image.split(".")[-1] == "nii" or args.image.split(".")[-2] == "nii":
        if not os.path.isfile(args.image):
            print("{}ERROR: \"{}{}{}\" does not exist!{}".format(
                bashColours.BOLDRED, bashColours.BOLDBLACK, args.image, bashColours.BOLDRED, bashColours.RESET))
            quit()
        imageReader = vtk.vtkNIFTIImageReader()
        imageReader.SetFileName(args.image)
        try:
            imageReader.Update()
            imageData = imageReader.GetOutput()
            # print(f"***NIFTI IMAGE***\n{imageData}\n\n")
        except:
            print(bashColours.BOLDRED, "ERROR reading the ", bashColours.BOLDBLACK, args.image, bashColours.BOLDRED, " file!\nAborting...", bashColours.RESET)
            quit()
    else:
        raise Exception(f"\"{args.image}\" does not have the expected extension!")

    # Correct Image origin and other stuff:
    if True:
        # print(imageData)
        imageData.SetOrigin((0.0, 0.0, 0.0))
        # imageData.SetDirectionMatrix((
        #     1.0, 0.0, 0.0,
        #     0.0, 1.0, 0.0,
        #     0.0, 0.0, 1.0))
        print(imageData)
        
    # Use numpy to process the mask:
    if not True:
        dimension = imageData.GetDimensions()
        image_flat_np_array = vtk_numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())
        image_np_array = image_flat_np_array.reshape(dimension)
        print("Image array, {}, min {}, median {}, mean {}, max {}".format(
            image_np_array.shape, image_np_array.min(), np.median(image_np_array), 
            image_np_array.mean(), image_np_array.max()))
        # Clean up:
        image_np_array = None
        image_flat_np_array = None

    # Instantiate MPR viewer:
    mpr = ThreePlaneView()
    mpr.SetImageDate(imageData)

    # Load Mask Data, if mask file is specified:
    maskData  = None
    if args.mask != "":
        if args.mask.split(".")[-1] == "vti":
            if not os.path.isfile(args.mask):
                print("{}ERROR: \"{}{}{}\" does not exist!{}".format(
                    bashColours.BOLDRED, bashColours.BOLDBLACK, vtiMaskFileName, bashColours.BOLDRED, bashColours.RESET))
                quit()
            maskReader = vtk.vtkXMLImageDataReader()
            maskReader.SetFileName(vtiMaskFileName)
            try:
                maskReader.Update()
                maskData = maskReader.GetOutput()
            except:
                print(bashColours.BOLDRED , "ERROR reading the " , bashColours.BOLDBLACK , vtiMaskFileName , bashColours.BOLDRED , " file!\nAborting..." , bashColours.RESET)
                quit()
        elif args.mask.split(".")[-1] == "nii" or args.mask.split(".")[-2] == "nii":
            if not os.path.isfile(args.mask):
                print("{}ERROR: \"{}{}{}\" does not exist!{}".format(
                    bashColours.BOLDRED,bashColours.BOLDBLACK,args.mask,bashColours.BOLDRED,bashColours.RESET))
                quit()
            maskReader = vtk.vtkNIFTIImageReader()
            maskReader.SetFileName(args.mask)
            try:
                maskReader.Update()
                maskData = maskReader.GetOutput()
                # print(f"***NIFTI MASK***\n{maskData}\n\n")
            except:
                print(bashColours.BOLDRED , "ERROR reading the " , bashColours.BOLDBLACK , args.mask , bashColours.BOLDRED , " file!\nAborting..." , bashColours.RESET)
                quit()
        elif args.mask.split(".")[-1] == "nrrd":
            if not os.path.isfile(args.mask):
                print("{}ERROR: \"{}{}{}\" does not exist!{}".format(
                    bashColours.BOLDRED, bashColours.BOLDBLACK, args.mask, bashColours.BOLDRED, bashColours.RESET))
                quit()
            maskReader = vtk.vtkNrrdReader()
            maskReader.SetFileName(args.mask)
            try:
                maskReader.Update()
                maskData = maskReader.GetOutput()
            except:
                print(bashColours.BOLDRED, "ERROR reading the ", bashColours.BOLDBLACK, args.mask, bashColours.BOLDRED, " file!\nAborting...", bashColours.RESET)
                quit()
        else:
            raise Exception(f"\"{args.mask}\" does not have the expected extension!")

    # Set mask colors if there is mask:
    if maskData is not None:
        if maskData.GetDimensions() != imageData.GetDimensions():
            print(bashColours.BOLDRED , "ERROR: Inconsistent number of dimensions between image " , bashColours.BOLDBLACK , imageData.GetDimensions() , bashColours.BOLDRED , " and mask" ,\
                bashColours.BOLDBLACK , maskData.GetDimensions()  , bashColours.BOLDRED , "!\nAborting..." , bashColours.RESET)
            quit()

        maskColoursList = args.color_map
        if len(maskColoursList) == 0:
            ## Set colors:
            alpha = 0.85
            maskColoursList.append([ 1.0, 0.0, 0.0, alpha])  #  1 - Red 
            maskColoursList.append([ 0.0, 1.0, 0.0, alpha])  #  2 - Green
            maskColoursList.append([ 0.0, 0.0, 1.0, alpha])  #  3 - Blue  
            maskColoursList.append([ 1.0, 1.0, 0.0, alpha])  #  4 - Yellow 
            maskColoursList.append([ 1.0, 0.0, 1.0, alpha])  #  5 - Magenta  
            maskColoursList.append([ 0.0, 1.0, 1.0, alpha])  #  6 - Cyan 
            maskColoursList.append([ 0.6, 0.6, 1.0, alpha])  #  7 - Light Purple 
            maskColoursList.append([ 1.0, 0.4, 0.0, alpha])  #  8 - Orange 
            maskColoursList.append([ 0.0, 0.0, 0.5, alpha])  #  9 - Navy 
            maskColoursList.append([ 0.5, 0.0, 0.0, alpha])  # 10 - Dark Red
            maskColoursList.append([ 0.0, 0.5, 0.0, alpha])  # 11 - Dark Green
            maskColoursList.append([ 1.0, 0.8, 0.6, alpha])  # 12 - Khaky
            maskColoursList.append([ 1.0, 0.6, 0.8, alpha])  # 13 - Pink 

            ## Use numpy to process the mask:
            dimension = maskData.GetDimensions()
            mask_flat_np_array = vtk_numpy_support.vtk_to_numpy(maskData.GetPointData().GetScalars())
            mask_flat_np_array = mask_flat_np_array.reshape(dimension)

            ## This is specific to Artrya masks: 
            if not True:
                print(f"Transposing Mask array: {mask_flat_np_array.shape} --> ", end="")
                mask_flat_np_array = mask_flat_np_array.transpose(2, 1, 0)
                mask_flat_np_array = np.flip(mask_flat_np_array, axis=2)
                print(f"{mask_flat_np_array.shape}")

            ## Make sure that voxel values are consistent with the number of colors:
            mask_flat_np_array = np.round(mask_flat_np_array)
            mask_flat_np_array = np.where(mask_flat_np_array <= 0, 0.0, mask_flat_np_array)
            mask_flat_np_array = np.where(mask_flat_np_array > 13, 0.0, mask_flat_np_array)
            vtk_data = vtk_numpy_support.numpy_to_vtk(
                num_array=mask_flat_np_array.flatten(), 
                deep=True, 
                array_type=vtk.VTK_FLOAT)
            maskData.GetPointData().SetScalars(vtk_data)

        # Correct mask origin and other stuff:
        if True:
            # print(maskData)
            maskData.SetOrigin(imageData.GetOrigin())
            # maskData.SetDirectionMatrix(imageData.GetDirectionMatrix())
            print(maskData)

        mpr.SetMaskData(maskData, maskColoursList)
        
        if maskData.GetSpacing() != imageData.GetSpacing():
            print("WARNING: Image and mask have different spacings!")
            print(f"{imageData.GetSpacing()} != {maskData.GetSpacing()}")
            print("Using image spacing...")
            maskData.SetSpacing(imageData.GetSpacing())

    # Set window level:
    if args.window_size == []:
        mpr.SetViewersWindowSize()
    elif len(args.window_size) == 1:
        mpr.SetViewersWindowSize(args.window_size[0], args.window_size[0])
    elif len(args.window_size) == 2:
        mpr.SetViewersWindowSize(args.window_size[0] , args.window_size[1])
    else:
        raise Exception(f"Window size argument is wrong: \"{args.window_size}\"")

    # Set background
    if args.background is None:
        mpr.SetViewersBackgroundColour()
    else:
        mpr.SetViewersBackgroundColour(args.background[0] , args.background[1] , args.background[2])

    # Set pixel interpolation
    if args.interpolation is None :
        mpr.SetInterpolation()
    else:
        mpr.SetInterpolation(args.interpolation)

    # Set Window width, level, and title
    mpr.Render()
    mpr.SetViewersWindowLevel(args.window_level, args.window_width) 
    mpr.SetViewersWindowName(args.window_title)
    mpr.Initialize()
    mpr.Start()

    # Close window:
    mpr.Finalize()
    mpr.TerminateApp()


if __name__ == "__main__":
    main()