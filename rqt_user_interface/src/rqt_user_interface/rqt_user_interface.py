#!/usr/bin/env python3
import os
import rospy
import rospkg

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from std_msgs.msg import String
from grasp.srv import pipeline_command, pipeline_commandResponse

class RqtPipeline(Plugin):

    def __init__(self, context):
        super(RqtPipeline, self).__init__(context)
        # Give QObjects reasonable names
        self.setObjectName('RqtPipeline')

        # Process standalone plugin command-line arguments
        from argparse import ArgumentParser
        parser = ArgumentParser()
        # Add argument(s) to the parser.
        parser.add_argument("-q", "--quiet", action="store_true",
                      dest="quiet",
                      help="Put plugin in silent mode")
        args, unknowns = parser.parse_known_args(context.argv())

        # Create QWidget
        self._widget = QWidget()
        # Get path to UI file which should be in the "resource" folder of this package
        ui_file = os.path.join(rospkg.RosPack().get_path('rqt_user_interface'), 'resource', 'rqt.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget)
        # Give QObjects reasonable names
        self._widget.setObjectName('RqtPipelineUI')
        # Show _widget.windowTitle on left-top of each plugin (when 
        # it's set in _widget). This is useful when you open multiple 
        # plugins at once. Also if you open multiple instances of your 
        # plugin at once, these lines add number to make it easy to 
        # tell from pane to pane
        if context.serial_number() > 1:
            self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))
        # Add widget to the user interface
        context.add_widget(self._widget)
        
        
        self.pipeline = None
        #Define functions for buttons here
        self._widget.CreateCrate.clicked[bool].connect(lambda: self.send_command_to_pipeline("create_crate"))
        self._widget.GoToSavedPose.clicked[bool].connect(lambda: self.send_command_to_pipeline("go_to_saved_pose"))
        self._widget.SavePose.clicked[bool].connect(lambda: self.send_command_to_pipeline("save_pose"))
        self._widget.SavePosePlace.clicked[bool].connect(lambda: self.send_command_to_pipeline("save_pose_place"))
        self._widget.DetectTrussOBB.clicked[bool].connect(lambda: self.send_command_to_pipeline("detect_truss_obb"))
        self._widget.DetectTrussManual.clicked[bool].connect(lambda: self.send_command_to_pipeline("detect_truss_manual"))
        self._widget.GoToTruss.clicked[bool].connect(lambda: self.send_command_to_pipeline("go_to_truss"))
        self._widget.CreateMap.clicked[bool].connect(lambda: self.send_command_to_pipeline("create_map"))
        self._widget.DetermineGraspCandidatesManual.clicked[bool].connect(lambda: self.send_command_to_pipeline("determine_grasp_candidates_manual"))
        self._widget.DetermineGraspCandidatesOrientedKeypoint.clicked[bool].connect(lambda: self.send_command_to_pipeline("determine_grasp_candidates_oriented_keypoint"))
        self._widget.ChooseGraspPoseFromCandidatesDepthImage.clicked[bool].connect(lambda: self.send_command_to_pipeline("choose_grasp_pose_from_candidates_depth_image"))
        self._widget.ChooseGraspPoseFromCandidatesRandom.clicked[bool].connect(lambda: self.send_command_to_pipeline("choose_grasp_pose_from_candidates_random"))
        self._widget.ChooseGraspPoseFromCandidatesCenter.clicked[bool].connect(lambda: self.send_command_to_pipeline("choose_grasp_pose_from_candidates_center"))
        self._widget.Grasp.clicked[bool].connect(lambda: self.send_command_to_pipeline("grasp"))
        self._widget.OpenGripper.clicked[bool].connect(lambda: self.send_command_to_pipeline("open_gripper"))
        self._widget.PreGraspGripper.clicked[bool].connect(lambda: self.send_command_to_pipeline("pre_grasp_gripper"))
        self._widget.CloseGripper.clicked[bool].connect(lambda: self.send_command_to_pipeline("close_gripper"))
        self._widget.ExecuteFullPipeline.clicked[bool].connect(lambda: self.send_command_to_pipeline("execute_full_pipeline"))
        self._widget.ExecuteFullPipelineCrate.clicked[bool].connect(lambda: self.send_command_to_pipeline("execute_full_pipeline_crate"))
        self._widget.ToggleExecuteFullPipelineRepeat.clicked[bool].connect(lambda: self.send_command_to_pipeline("toggle_execute_full_pipeline_repeat"))
    
    def send_command_to_pipeline(self, command):
        if self.pipeline == None:
            rospy.wait_for_service('rqt_service', timeout=10)
            self.pipeline_service = rospy.ServiceProxy('rqt_service', pipeline_command)
        try:
            if command != None:
                self.pipeline_service(command)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))

    def shutdown_plugin(self):
        # TODO unregister all publishers here
        pass

    def save_settings(self, plugin_settings, instance_settings):
        # TODO save intrinsic configuration, usually using:
        # instance_settings.set_value(k, v)
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        # TODO restore intrinsic configuration, usually using:
        # v = instance_settings.value(k)
        pass

    #def trigger_configuration(self):
        # Comment in to signal that the plugin has a way to configure
        # This will enable a setting button (gear icon) in each dock widget title bar
        # Usually used to open a modal configuration dialog
