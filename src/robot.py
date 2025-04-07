import pybullet as p
import numpy as np
import math
import os

# Constants
NOL = 6  # Number of legs
NOJ = 3  # Number of joints per leg

class Robot:
    """Hexapod robot model using URDF"""
    
    def __init__(self):
        self.body = None
        self.joint_map = {}
        self.leg_joint_ids = [[None for _ in range(NOJ)] for _ in range(NOL)]
        self.fixed = None
    
    def make_robot(self, physics_client):
        """Load the robot from URDF file"""
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level and then to models directory
        urdf_path = os.path.join(os.path.dirname(current_dir), 'models', 'hexapod.urdf')
        
        print(f"Loading robot from: {urdf_path}")
        
        try:
            # Load the URDF file
            self.body = p.loadURDF(
                urdf_path,
                basePosition=[0.0, 0.0, 0.5],
                useFixedBase=False,
                physicsClientId=physics_client
            )
            
            # Map joint names to IDs
            self._build_joint_map(physics_client)
            
            # Create fixed constraint to hold robot in the air initially
            self.fixed = p.createConstraint(
                parentBodyUniqueId=self.body,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0.5],
                physicsClientId=physics_client
            )
            
            return True
        except Exception as e:
            print(f"Error loading robot: {e}")
            return False
    
    def _build_joint_map(self, physics_client):
        """Build mapping between joint names and IDs"""
        num_joints = p.getNumJoints(self.body, physicsClientId=physics_client)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.body, i, physicsClientId=physics_client)
            joint_name = joint_info[1].decode('utf-8')
            self.joint_map[joint_name] = i
            
            # Map leg joints to our array structure
            for leg in range(NOL):
                for joint in range(NOJ):
                    if joint_name == f"leg{leg}_joint{joint+1}":
                        self.leg_joint_ids[leg][joint] = i
        
        print(f"Found {len(self.joint_map)} joints in the robot")

# Create a singleton instance
robot = Robot()

def p_control(joint_id, target, physics_client):
    """Proportional control for joint movement"""
    kp = 5.0
    fmax = 20.0
    
    # Get current angle
    current_angle = p.getJointState(robot.body, joint_id, physicsClientId=physics_client)[0]
    diff = target - current_angle
    u = kp * diff
    
    p.setJointMotorControl2(
        bodyUniqueId=robot.body,
        jointIndex=joint_id,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=u,
        force=fmax,
        physicsClientId=physics_client
    )

def move_robot(tang, vel_counter, times, physics_client):
    """Move the robot according to target joint angles"""
    samstep = 20  # Sampling steps for feedback control
    
    vel_counter += 1
    
    # Apply controls to each joint
    for i in range(NOL):
        for j in range(NOJ):
            if robot.leg_joint_ids[i][j] is not None:
                p_control(robot.leg_joint_ids[i][j], tang[i][j], physics_client)
    
    # Update target joint angles periodically
    if vel_counter % samstep == 0:
        from genetic import loco_main
        times = loco_main(tang, times)
        vel_counter = 0
    
    return vel_counter, times

def release_robot(physics_client):
    """Release the robot from fixed constraint"""
    p.removeConstraint(robot.fixed, physicsClientId=physics_client)

def get_base_position_and_orientation(physics_client):
    """Get the robot's base position and orientation"""
    return p.getBasePositionAndOrientation(robot.body, physicsClientId=physics_client)

def get_base_rotation(physics_client):
    """Get the rotation matrix of the robot's base"""
    _, orientation = p.getBasePositionAndOrientation(robot.body, physicsClientId=physics_client)
    return p.getMatrixFromQuaternion(orientation)