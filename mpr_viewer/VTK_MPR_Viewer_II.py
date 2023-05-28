"""A stand alone utility for viewing NIFTI images.
"""

import sys
import os.path
import json

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_numpy_support
import argparse


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
            self.imageViewer.render()


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

    def update_cursor_position(self, centre=(0.0, 0.0, 0.0)):
        tolerance = 0.05
        if self.imageViewer.GetSliceOrientation() == self.imageViewer.SLICE_ORIENTATION_YZ:
           self.cursor.SetFocalPoint(centre[0]+tolerance,centre[1],centre[2])
        elif self.imageViewer.GetSliceOrientation() == self.imageViewer.SLICE_ORIENTATION_XZ:
             self.cursor.SetFocalPoint(centre[0],centre[1]+tolerance,centre[2])
        elif self.imageViewer.GetSliceOrientation() == self.imageViewer.SLICE_ORIENTATION_XY:
             self.cursor.SetFocalPoint(centre[0],centre[1],centre[2]+tolerance)
        else:
            print("ERROR!")    
        self.cursor.Update()

    def update_cursor_size(self, size=(1.0, 1.0, 1.0)):
        _bounds = self.cursor.GetModelBounds()
        _centre = (.5*(_bounds[1]+_bounds[0]) ,  .5*(_bounds[3]+_bounds[2]) ,  .5*(_bounds[5]+_bounds[4]))
        self.cursor.SetModelBounds(_centre[0]-_size[0] , centre[0]+_size[0],
                                    _centre[1]-_size[1] , centre[1]+_size[1],
                                    _centre[2]-_size[2] , centre[2]+_size[2])
        self.cursor.Update()

    def cursor_visibility(self):
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

    def set_image_viewer(self, mpr, imViewer):
        self.threePlaneView = mpr
        self.imageViewer = imViewer
        #
        self.imageViewer.GetInteractorStyle().RemoveAllObservers()

    def initialize(self):
        self.minSlice = self.imageViewer.GetSliceMin()
        self.maxSlice = self.imageViewer.GetSliceMax()

    def KeyPress(self, obj, event):
        key = obj.GetKeySym()
        if key == "Up":
            self.threePlaneView.dispatch_arrow_key_update(
                self.imageViewer, (0, self._cursor_move_step))
        elif key == "Down":
            self.threePlaneView.dispatch_arrow_key_update(
                self.imageViewer, (0, -self._cursor_move_step))
        elif key == "Left":
            self.threePlaneView.dispatch_arrow_key_update(
                self.imageViewer, (-self._cursor_move_step, 0))
        elif key == "Right":
            self.threePlaneView.dispatch_arrow_key_update(
                self.imageViewer, (self._cursor_move_step, 0))
        elif key == "f" or key == "F":  # Move cursor 10 steps!
            self._cursor_move_step = 10

    def KeyRelease(self, obj, event):
        key = obj.GetKeySym()
        if key == "r" or key == "R":
            self.threePlaneView.dispatch_window_level_reset(True, True)
        elif key == "w" or key == "W":
            self.threePlaneView.dispatch_window_level_reset(True, False)
        elif key == "l" or key == "L":
            self.threePlaneView.dispatch_window_level_reset(False, True)
        elif key == "s" or key == "S":
            self.threePlaneView.take_screenshots()
        elif key == "q" or key == "Q":
            self.threePlaneView.finalize()
            self.threePlaneView.terminate_app()
        elif key == "c" or key == "C":
            self.threePlaneView.change_curser_visibility()
        elif key == "f" or key == "F":  # Move cursor 10 steps!
            self._cursor_move_step = 1
        elif key == "period":  
            slice = self.imageViewer.GetSlice() + 1
            if slice >= self.minSlice and slice <= self.maxSlice:
                self.threePlaneView.dispatch_slice_update(self.imageViewer, slice)
        elif key == "comma":  
            slice = self.imageViewer.GetSlice() - 1
            if slice >= self.minSlice and slice <= self.maxSlice:
                self.threePlaneView.dispatch_slice_update(self.imageViewer, slice)
        elif key == "h" or key == "H":
            print("Camera: ", self.imageViewer.GetRenderer().GetActiveCamera()) 

    def LeftButtonPress(self, obj, event):
        if self.initial_event_position is None:
            self.initial_event_position = obj.GetEventPosition()

    def LeftButtonRelease(self, obj, event):
        clickPos = obj.GetEventPosition()
        self.initial_event_position = None
        self.threePlaneView.refresh_current_window_level()

    def MouseMove(self, obj, event):
        current_event_position = obj.GetEventPosition()
        # print(f" -> {current_event_position}")
        if(self.initial_event_position is not None):
            self.threePlaneView.dispatch_window_level_event(
                current_event_position,
                self.initial_event_position)
        else:
            self.threePlaneView.dispatch_mouse_move(
                self.imageViewer,
                current_event_position)

    def MouseWheelForward(self, obj, event):
        slice = self.imageViewer.GetSlice() + 1
        if slice >= self.minSlice and slice <= self.maxSlice:
            self.threePlaneView.dispatch_slice_update(self.imageViewer, slice)

    def MouseWheelBackward(self, obj, event):
        slice = self.imageViewer.GetSlice() - 1
        if slice >= self.minSlice and slice <= self.maxSlice:
            self.threePlaneView.dispatch_slice_update(self.imageViewer, slice) 


class ThreePlaneView():
    def __init__(self, image_data, cursor_off=False):
        self._cursor_off = cursor_off
        self._image_spacing = None
        self._image_dimensions = None
        self.lastImageCoordinates = [0, 0, 0]
        self.position = [0.0, 0.0, 0.0]
        self.initial_window_width = None
        self.initial_window_level = None
        self.current_window_width = None
        self.current_window_level = None

        ## Create the three views: Axial Sagittal and Coronal
        self._view_x = vtk.vtkImageViewer2()
        self._view_y = vtk.vtkImageViewer2()
        self._view_z = vtk.vtkImageViewer2()  
        
        self._image_data = image_data
        self._set_slice_orientations()
        self._set_render_window_interactors()
        self._set_custom_interactor_managers()
        self._set_text_props()
        self._set_cursors()
        self._set_image_date()
        self._update_cameras()
        
        ## Mask Actors:
        self.maskActorX = None
        self.maskActorY = None
        self.maskActorZ = None
        
        ## Pickers:
        self.pickerX = vtk.vtkPropPicker()
        self.pickerY = vtk.vtkPropPicker()
        self.pickerZ = vtk.vtkPropPicker()

    def _set_slice_orientations(self,):
        ## Set their corresponding orientations
        self._view_x.SetSliceOrientationToYZ()
        self._view_y.SetSliceOrientationToXZ()
        self._view_z.SetSliceOrientationToXY()

    def _set_image_date(self,):
        self._view_x.SetInputData(self._image_data)
        self._view_y.SetInputData(self._image_data)
        self._view_z.SetInputData(self._image_data)
        #
        self._interactor_mgr_x.initialize()
        self._interactor_mgr_y.initialize()
        self._interactor_mgr_z.initialize()
        #
        self._image_spacing    = self._image_data.GetSpacing()
        self._image_dimensions = self._image_data.GetDimensions()

    def _set_render_window_interactors(self,):
        ## Set Render Window Interactor
        self._render_window_interactor_x = vtk.vtkRenderWindowInteractor()
        self._render_window_interactor_y = vtk.vtkRenderWindowInteractor()
        self._render_window_interactor_z = vtk.vtkRenderWindowInteractor()
        #
        self._render_window_interactor_x.SetRenderWindow(self._view_x.GetRenderWindow())
        self._render_window_interactor_y.SetRenderWindow(self._view_y.GetRenderWindow())
        self._render_window_interactor_z.SetRenderWindow(self._view_z.GetRenderWindow())
        #
        self._view_x.SetupInteractor(self._render_window_interactor_x)
        self._view_y.SetupInteractor(self._render_window_interactor_y)
        self._view_z.SetupInteractor(self._render_window_interactor_z)

    def _set_custom_interactor_managers(self,):
        ## Set Custom Interactors: 
        self._interactor_mgr_x = CustomInteractorManager(self._render_window_interactor_x)
        self._interactor_mgr_y = CustomInteractorManager(self._render_window_interactor_y)
        self._interactor_mgr_z = CustomInteractorManager(self._render_window_interactor_z)
        #
        self._interactor_mgr_x.set_image_viewer(self, self._view_x)
        self._interactor_mgr_y.set_image_viewer(self, self._view_y)
        self._interactor_mgr_z.set_image_viewer(self, self._view_z)

    def _update_cameras(self,):
        ## Adjust Camera for each viewer:
        self.cameraX = self._view_x.GetRenderer().GetActiveCamera()
        self.cameraX.SetFocalPoint(0, 0, 0)
        self.cameraX.SetPosition(1, 0, 0)
        self.cameraX.SetViewUp(0, 0, -1)
        self.cameraX.SetParallelScale(128.0)
        # self.cameraX.Zoom(1.0)
        #
        self.cameraY = self._view_y.GetRenderer().GetActiveCamera()
        self.cameraY.SetFocalPoint(0, 0, 0)
        self.cameraY.SetPosition(0, 1, 0)
        self.cameraY.SetViewUp(0, 0, -1)
        self.cameraY.SetParallelScale(128.0)
        # self.cameraY.Zoom(1.0)
        #
        self.cameraZ = self._view_z.GetRenderer().GetActiveCamera()
        self.cameraZ.SetFocalPoint(0, 0, 0)
        self.cameraZ.SetPosition(0, 0, 1)
        self.cameraZ.SetViewUp(0, 1, 0)
        self.cameraZ.SetParallelScale(128.0)
        # self.cameraZ.Zoom(1.0)

    def _set_text_props(self,):
        ## Text Props
        self._text_prop_x = TextProp(self._view_x)
        self._text_prop_y = TextProp(self._view_y)
        self._text_prop_z = TextProp(self._view_z)

    def _set_cursors(self,):
        ## Set Cursor:
        if self._cursor_off:
            self._cursor_x = None
            self._cursor_y = None
            self._cursor_z = None
        else:
            self._cursor_x = Cursor3D(self._view_x)
            self._cursor_y = Cursor3D(self._view_y)
            self._cursor_z = Cursor3D(self._view_z)

    def set_viewers_window_name(self,window_title=" ¯\\_(ツ)_/¯"):
        self._view_x.GetRenderWindow().SetWindowName(" ".join([window_title,"(X)"]))
        self._view_y.GetRenderWindow().SetWindowName(" ".join([window_title,"(Y)"]))
        self._view_z.GetRenderWindow().SetWindowName(" ".join([window_title,"(Z)"]))

    def set_viewers_window_size(self, x_pixels=1024, y_pixels=1024):
        self._render_window_interactor_x.GetRenderWindow().SetSize(x_pixels, y_pixels)
        self._render_window_interactor_y.GetRenderWindow().SetSize(x_pixels, y_pixels)
        self._render_window_interactor_z.GetRenderWindow().SetSize(x_pixels, y_pixels)

    def set_viewers_background_color(self, r=0.125, g=0.0, b=0.125):
        self._view_x.GetRenderer().SetBackground(r, g, b)
        self._view_y.GetRenderer().SetBackground(r, g, b)
        self._view_z.GetRenderer().SetBackground(r, g, b)

    def set_viewers_window_level(self, level, width):
        if (width is not None) and (level is not None):
            self.initial_window_width = width
            self.initial_window_level = level
            #
            self._view_x.SetColorLevel(level)
            self._view_y.SetColorLevel(level)
            self._view_z.SetColorLevel(level)
            #
            self._view_x.SetColorWindow(width)
            self._view_y.SetColorWindow(width)
            self._view_z.SetColorWindow(width)
        else:
            self.initial_window_width = self._view_x.GetColorWindow()
            self.initial_window_level = self._view_x.GetColorLevel()
        #
        self.current_window_width = self.initial_window_width
        self.current_window_level = self.initial_window_level

    def set_interpolation(self,interpolationType="Nearest"):
        if interpolationType == "Cubic":
            self._view_x.GetImageActor().GetProperty().SetInterpolationTypeToCubic()
            self._view_y.GetImageActor().GetProperty().SetInterpolationTypeToCubic()
            self._view_z.GetImageActor().GetProperty().SetInterpolationTypeToCubic()
        elif interpolationType == "Linear":
            self._view_x.GetImageActor().GetProperty().SetInterpolationTypeToLinear()
            self._view_y.GetImageActor().GetProperty().SetInterpolationTypeToLinear()
            self._view_z.GetImageActor().GetProperty().SetInterpolationTypeToLinear()
        else:
            self._view_x.GetImageActor().GetProperty().SetInterpolationTypeToNearest()
            self._view_y.GetImageActor().GetProperty().SetInterpolationTypeToNearest()
            self._view_z.GetImageActor().GetProperty().SetInterpolationTypeToNearest()

    def set_mask_data(self, mask_Data, colours_List):
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
        self._view_x.GetRenderer().AddActor(self.maskActorX)
        self._view_y.GetRenderer().AddActor(self.maskActorY)
        self._view_z.GetRenderer().AddActor(self.maskActorZ)
        #
        self.render()

    def update_masks(self,currentImageViewer=None):
        if self.maskActorX is not None and \
           self.maskActorY is not None and \
           self.maskActorZ is not None :
            if currentImageViewer is not None:
                self.maskActorX.SetDisplayExtent(self._view_x.GetImageActor().GetDisplayExtent())
                self.maskActorY.SetDisplayExtent(self._view_y.GetImageActor().GetDisplayExtent())
                self.maskActorZ.SetDisplayExtent(self._view_z.GetImageActor().GetDisplayExtent())
            self.maskActorX.Update()
            self.maskActorY.Update()
            self.maskActorZ.Update()

    def dispatch_mouse_move(self, imViewer, eventPos):
        if imViewer == self._view_x:
            self.pickerX.Pick(eventPos[0], eventPos[1], 0, self._view_x.GetRenderer())
            if self.pickerX.GetPath():
                position = self.pickerX.GetPickPosition()
                image_coordinate = (self._view_x.GetSlice() - self._view_x.GetSliceMin(),
                                    round(position[1] / self._image_spacing[1]),
                                    round(position[2] / self._image_spacing[2]))
                self.position = [
                    self._view_x.GetSlice() * self._image_spacing[0], position[1], position[2] ]
                self._view_y.SetSlice(image_coordinate[1])
                self._view_z.SetSlice(image_coordinate[2])
            else:
                image_coordinate = (-1 , -1 , -1)
            # print(f"---> {image_coordinate}")
        elif imViewer == self._view_y:
            self.pickerY.Pick(eventPos[0], eventPos[1], 0, self._view_y.GetRenderer())
            if self.pickerY.GetPath():
                position = self.pickerY.GetPickPosition()
                image_coordinate = (round(position[0] / self._image_spacing[0]),
                                    self._view_y.GetSlice() - self._view_y.GetSliceMin(),
                                    round(position[2] / self._image_spacing[2]))
                self.position = [
                    position[0] , self._view_y.GetSlice() * self._image_spacing[1], position[2]]
                self._view_x.SetSlice(image_coordinate[0])
                self._view_z.SetSlice(image_coordinate[2])
            else:
                image_coordinate = (-1 , -1 , -1)
            # print(f"---> {image_coordinate}")
        elif imViewer == self._view_z:
            self.pickerZ.Pick(eventPos[0], eventPos[1], 0, self._view_z.GetRenderer())
            if self.pickerZ.GetPath():
                position = self.pickerZ.GetPickPosition()
                image_coordinate = (round(position[0] / self._image_spacing[0]),
                                    round(position[1] / self._image_spacing[1]),
                                    self._view_z.GetSlice() - self._view_z.GetSliceMin())
                self.position = [
                    position[0], position[1], self._view_z.GetSlice() * self._image_spacing[2]]
                self._view_x.SetSlice(image_coordinate[0])
                self._view_y.SetSlice(image_coordinate[1])
            else:
                image_coordinate = (-1 , -1 , -1)
            # print(f"---> {image_coordinate}")
        else:
            print("ERROR: Unspecified vtkImageViewer!")
            return
        # 
        if image_coordinate[0] >= 0 and image_coordinate[0] < self._image_dimensions[0] and\
           image_coordinate[1] >= 0 and image_coordinate[1] < self._image_dimensions[1] and\
           image_coordinate[2] >= 0 and image_coordinate[2] < self._image_dimensions[2]:
            voxVal = self._view_x.GetInput().GetScalarComponentAsDouble(
                image_coordinate[0] , image_coordinate[1] , image_coordinate[2] , 0)
            self._text_prop_x.UpdateTextProp("".join(
                [ "(" , str(image_coordinate[0]+1) , "/" , str(self._image_dimensions[0]) , " , " ,\
                        str(image_coordinate[1]+1) , "/" , str(self._image_dimensions[1]) , " , " ,\
                        str(image_coordinate[2]+1) , "/" , str(self._image_dimensions[2]) , ")    " , str(voxVal) ]))
            self._text_prop_y.UpdateTextProp("".join(
                [ "(" , str(image_coordinate[0]+1) , "/" , str(self._image_dimensions[0]) , " , " ,\
                        str(image_coordinate[1]+1) , "/" , str(self._image_dimensions[1]) , " , " ,\
                        str(image_coordinate[2]+1) , "/" , str(self._image_dimensions[2]) , ")    " , str(voxVal) ]))
            self._text_prop_z.UpdateTextProp("".join(
                [ "(" , str(image_coordinate[0]+1) , "/" , str(self._image_dimensions[0]) , " , " ,\
                        str(image_coordinate[1]+1) , "/" , str(self._image_dimensions[1]) , " , " ,\
                        str(image_coordinate[2]+1) , "/" , str(self._image_dimensions[2]) , ")    " , str(voxVal) ]))
            #
            self.lastImageCoordinates = list(image_coordinate)
            #
            if not self._cursor_off:
                self._cursor_x.update_cursor_position(self.position)
                self._cursor_y.update_cursor_position(self.position)
                self._cursor_z.update_cursor_position(self.position)
        else:
            self._text_prop_x.UpdateTextProp("--")
            self._text_prop_y.UpdateTextProp("--")
            self._text_prop_z.UpdateTextProp("--")
        #
        self.update_masks(imViewer)
        self.render()

    def dispatch_arrow_key_update(self, imViewer, move=(0, 0)):
        if imViewer == self._view_x:
            self._view_y.SetSlice(self._view_y.GetSlice() - move[0])
            self._view_z.SetSlice(self._view_z.GetSlice() - move[1])
            self.lastImageCoordinates[1] = self._view_y.GetSlice() - self._view_y.GetSliceMin()
            self.lastImageCoordinates[2] = self._view_z.GetSlice() - self._view_z.GetSliceMin()
        elif imViewer == self._view_y:
            self._view_x.SetSlice(self._view_x.GetSlice() + move[0])
            self._view_z.SetSlice(self._view_z.GetSlice() - move[1])
            self.lastImageCoordinates[0] = self._view_x.GetSlice() - self._view_x.GetSliceMin()
            self.lastImageCoordinates[2] = self._view_z.GetSlice() - self._view_z.GetSliceMin()
        elif imViewer == self._view_z:
            self._view_x.SetSlice(self._view_x.GetSlice() + move[0])
            self._view_y.SetSlice(self._view_y.GetSlice() + move[1])
            self.lastImageCoordinates[0] = self._view_x.GetSlice() - self._view_x.GetSliceMin()
            self.lastImageCoordinates[1] = self._view_y.GetSlice() - self._view_y.GetSliceMin()
        else:
            print("ERROR: Unspecified vtkImageViewer!")
            return

        self.position[0] = self._view_x.GetSlice() * self._image_spacing[0]
        self.position[1] = self._view_y.GetSlice() * self._image_spacing[1]
        self.position[2] = self._view_z.GetSlice() * self._image_spacing[2]
        #
        voxVal = self._view_x.GetInput().GetScalarComponentAsDouble(
            self.lastImageCoordinates[0], 
            self.lastImageCoordinates[1], 
            self.lastImageCoordinates[2], 
            0)
        self._text_prop_x.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self._image_dimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self._image_dimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self._image_dimensions[2]) , ")    " , str(voxVal) ]))
        self._text_prop_y.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self._image_dimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self._image_dimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self._image_dimensions[2]) , ")    " , str(voxVal) ]))
        self._text_prop_z.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self._image_dimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self._image_dimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self._image_dimensions[2]) , ")    " , str(voxVal) ]))
        #
        if not self._cursor_off:
            self._cursor_x.update_cursor_position(self.position)
            self._cursor_y.update_cursor_position(self.position)
            self._cursor_z.update_cursor_position(self.position)
        #
        self.update_masks(imViewer)
        self.render()    

    def dispatch_slice_update(self, imViewer, slice):
        if imViewer == self._view_x:
            self._view_x.SetSlice(slice)
            self.lastImageCoordinates[0] = slice - self._view_x.GetSliceMin()
            self.position[0] = self._view_x.GetSlice() * self._image_spacing[0]
        elif imViewer == self._view_y:
            self._view_y.SetSlice(slice)
            self.lastImageCoordinates[1] = slice - self._view_y.GetSliceMin()
            self.position[1] = self._view_y.GetSlice() * self._image_spacing[1]
        elif imViewer == self._view_z:
            self._view_z.SetSlice(slice)
            self.lastImageCoordinates[2] = slice - self._view_z.GetSliceMin()
            self.position[2] = self._view_z.GetSlice() * self._image_spacing[2]
        else:
            print("ERROR: Unspecified vtkImageViewer!")
            return
        #
        voxVal = self._view_x.GetInput().GetScalarComponentAsDouble(
            self.lastImageCoordinates[0] , self.lastImageCoordinates[1] , self.lastImageCoordinates[2] , 0)
        self._text_prop_x.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self._image_dimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self._image_dimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self._image_dimensions[2]) , ")    " , str(voxVal) ]))
        self._text_prop_y.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self._image_dimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self._image_dimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self._image_dimensions[2]) , ")    " , str(voxVal) ]))
        self._text_prop_z.UpdateTextProp("".join(
            [ "(" , str(self.lastImageCoordinates[0] + 1) , "/" , str(self._image_dimensions[0]) , " , " ,\
                    str(self.lastImageCoordinates[1] + 1) , "/" , str(self._image_dimensions[1]) , " , " ,\
                    str(self.lastImageCoordinates[2] + 1) , "/" , str(self._image_dimensions[2]) , ")    " , str(voxVal) ]))
        #
        if not self._cursor_off:
            self._cursor_x.update_cursor_position(self.position)
            self._cursor_y.update_cursor_position(self.position)
            self._cursor_z.update_cursor_position(self.position)
        #
        self.update_masks(imViewer)
        self.render()

    def refresh_current_window_level(self):
        self.current_window_level = self._view_x.GetColorLevel()
        self.current_window_width = self._view_x.GetColorWindow()

    def dispatch_window_level_event(self,current_dispatched_event_position,initial_dispatched_event_positions):
        level = self.current_window_level + round(
            (current_dispatched_event_position[0] - initial_dispatched_event_positions[0]) / 0.25)
        width = self.current_window_width + round(
            (current_dispatched_event_position[1] - initial_dispatched_event_positions[1]) / 0.25)
        # width = max(self.current_window_width + round(
        #    (current_dispatched_event_position[1]-initial_dispatched_event_positions[1])/.25) , 10.)
        #
        self._view_x.SetColorLevel(level)
        self._view_y.SetColorLevel(level)
        self._view_z.SetColorLevel(level)
        #
        self._view_x.SetColorWindow(width)
        self._view_y.SetColorWindow(width)
        self._view_z.SetColorWindow(width)
        #
        self.render()

    def dispatch_window_level_reset(self,reset_window,reset_level):
        if reset_window:
            self._view_x.SetColorWindow(self.initial_window_width)
            self._view_y.SetColorWindow(self.initial_window_width)
            self._view_z.SetColorWindow(self.initial_window_width)
        #
        if reset_level :
            self._view_x.SetColorLevel(self.initial_window_level)
            self._view_y.SetColorLevel(self.initial_window_level)
            self._view_z.SetColorLevel(self.initial_window_level)
        #
        self.refresh_current_window_level()
        self.render()

    def take_screenshots(self,):
        ## Screenshot PNGImageWriter and WindowToImageFilter:
        screenshotImageWRiterX = vtk.vtkPNGWriter()
        screenshotImageWRiterY = vtk.vtkPNGWriter()
        screenshotImageWRiterZ = vtk.vtkPNGWriter()
        #
        screenshotX = vtk.vtkWindowToImageFilter()
        screenshotY = vtk.vtkWindowToImageFilter()
        screenshotZ = vtk.vtkWindowToImageFilter()
        #
        screenshotX.SetInput(self._view_x.GetRenderWindow())
        screenshotY.SetInput(self._view_y.GetRenderWindow())
        screenshotZ.SetInput(self._view_z.GetRenderWindow())
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
            [ "./" , str(self._view_x.GetRenderWindow().GetWindowName()).split(" (X)")[0] , 
              "__X-" , str(self._view_x.GetSlice()) , ".png" ])
        temp_filename_y = "".join(
            [ "./" , str(self._view_y.GetRenderWindow().GetWindowName()).split(" (Y)")[0] , 
              "__Y-" , str(self._view_y.GetSlice()) , ".png" ])
        temp_filename_z = "".join(
            [ "./" , str(self._view_z.GetRenderWindow().GetWindowName()).split(" (Z)")[0] , 
              "__Z-" , str(self._view_z.GetSlice()) , ".png" ])
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

    def change_curser_visibility(self):
        if not self._cursor_off:
            self._cursor_x.cursor_visibility()
            self._cursor_y.cursor_visibility()
            self._cursor_z.cursor_visibility()
        self.render()


    def initialize(self):
        self._render_window_interactor_x.Initialize()
        self._render_window_interactor_y.Initialize()
        self._render_window_interactor_z.Initialize()

    def render(self):
        self._view_x.Render()
        self._view_y.Render()
        self._view_z.Render()

    def start(self):
        self._render_window_interactor_x.Start()
        self._render_window_interactor_y.Start()
        self._render_window_interactor_z.Start()
        
    def finalize(self):
        self._render_window_interactor_x.GetRenderWindow().Finalize()
        self._render_window_interactor_y.GetRenderWindow().Finalize()
        self._render_window_interactor_z.GetRenderWindow().Finalize()

    def terminate_app(self):
        self._render_window_interactor_x.TerminateApp()
        self._render_window_interactor_y.TerminateApp()
        self._render_window_interactor_z.TerminateApp()
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

    # Instantiate MPR viewer:
    mpr = ThreePlaneView(imageData, cursor_off=False)

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

        mpr.set_mask_data(maskData, maskColoursList)
        
        if maskData.GetSpacing() != imageData.GetSpacing():
            print("WARNING: Image and mask have different spacings!")
            print(f"{imageData.GetSpacing()} != {maskData.GetSpacing()}")
            print("Using image spacing...")
            maskData.SetSpacing(imageData.GetSpacing())
        
    # Export 
    export_masked_image_to_npy = True
    if export_masked_image_to_npy and maskData is not None:
        #
        ## Check dimensions:
        image_dims = imageData.GetDimensions()
        mask_dims = maskData.GetDimensions()
        assert(image_dims == mask_dims)
        #
        ## Get image and mask as numpy arrays
        image_flat_np_array = vtk_numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())
        image_np_array = image_flat_np_array.reshape(image_dims)
        mask_flat_np_array = vtk_numpy_support.vtk_to_numpy(maskData.GetPointData().GetScalars())
        mask_np_array = mask_flat_np_array.reshape(mask_dims)
        #
        ## Mask image and export to NumPy file:
        masked_image = np.where(mask_np_array > 0, image_np_array, -1024.0)
        np.save("./masked_image.npy", masked_image)
        #
        ## Clean up:
        image_np_array = None
        image_flat_np_array = None
        mask_np_array = None
        mask_flat_np_array = None
        masked_image = None

    # Set window level:
    if args.window_size == []:
        mpr.set_viewers_window_size()
    elif len(args.window_size) == 1:
        mpr.set_viewers_window_size(args.window_size[0], args.window_size[0])
    elif len(args.window_size) == 2:
        mpr.set_viewers_window_size(args.window_size[0] , args.window_size[1])
    else:
        raise Exception(f"Window size argument is wrong: \"{args.window_size}\"")

    # Set background
    if args.background is None:
        mpr.set_viewers_background_color()
    else:
        mpr.set_viewers_background_color(args.background[0] , args.background[1] , args.background[2])

    # Set pixel interpolation
    if args.interpolation is None :
        mpr.set_interpolation()
    else:
        mpr.set_interpolation(args.interpolation)

    # Set Window width, level, and title
    mpr.set_viewers_window_level(args.window_level, args.window_width) 
    mpr.set_viewers_window_name(args.window_title)
    mpr.initialize()
    mpr.render()
    mpr.start()

    # Close window:
    mpr.finalize()
    mpr.terminate_app()


if __name__ == "__main__":
    main()