import pybullet as p
import numpy as np
import math

# Constants
NOL = 6  # Number of legs
NOJ = 3  # Number of joints per leg

class LegModel:
    """Model of a single leg with joints"""
    
    def __init__(self):
        self.sides = [0.1, 0.2, 0.1]  # width, length, height
        self.mass = 0.05
        self.rest = 0.04  # joint spacing
        self.geom = [None] * (NOJ + 1)
        self.body = [None] * (NOJ + 1)
        self.joint = [None] * (NOJ + 1)
    
    def make_leg(self, physics_client):
        """Create a leg in the physics simulation"""
        for i in range(NOJ + 1):
            # Create visual shape
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[self.sides[0]/2, self.sides[1]/2, self.sides[2]/2],
                rgbaColor=[0.2, 0.4, 0.2, 1],
                physicsClientId=physics_client
            )
            
            # Create collision shape
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[self.sides[0]/2, self.sides[1]/2, self.sides[2]/2],
                physicsClientId=physics_client
            )
            
            # Create body
            y_pos = self.sides[1] * 0.5 + (self.sides[1] + self.rest) * i
            self.body[i] = p.createMultiBody(
                baseMass=self.mass,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[0.0, y_pos, 0.0],
                physicsClientId=physics_client
            )
            
            # Create joints to connect body segments
            if i > 0:
                hinge_pos = [(self.sides[1] + self.rest) * i - self.rest * 0.5]
                if i == 1:
                    joint_axis = [0, 1, 0]  # Y-axis rotation for first joint
                else:
                    joint_axis = [1, 0, 0]  # X-axis rotation for other joints
                
                self.joint[i] = p.createConstraint(
                    parentBodyUniqueId=self.body[i-1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=self.body[i],
                    childLinkIndex=-1,
                    jointType=p.JOINT_REVOLUTE,
                    jointAxis=joint_axis,
                    parentFramePosition=[0, self.sides[1]/2 + self.rest/2, 0],
                    childFramePosition=[0, -self.sides[1]/2, 0],
                    physicsClientId=physics_client
                )
    
    def translate(self, pos, R, physics_client):
        """Move the leg to a new position with rotation"""
        for i in range(NOJ + 1):
            # Get current position
            curr_pos, _ = p.getBasePositionAndOrientation(self.body[i], physicsClientId=physics_client)
            
            # Apply rotation matrix R to current position
            R_matrix = np.array(R).reshape(3, 4)[:3, :3]  # Extract 3x3 rotation matrix
            new_pos = R_matrix @ np.array(curr_pos)
            
            # Add translation
            new_pos = new_pos + np.array(pos)
            
            # Set new position and orientation
            p.resetBasePositionAndOrientation(
                self.body[i],
                new_pos.tolist(),
                p.getQuaternionFromMatrix(R_matrix.tolist()),
                physicsClientId=physics_client
            )

class HeadModel:
    """Model of the robot's head with eyes"""
    
    def __init__(self):
        self.geom = None
        self.body = None
        self.eye_g = [None, None]
        self.eye_b = [None, None]
        self.sides = [0.2, 0.2, 0.1]  # length, width, height
        self.radius = 0.05  # eye radius
        self.pos = [
            [self.sides[0]*0.5, self.sides[1]*0.5, self.sides[2]*0.5],
            [self.sides[0]*0.5, -self.sides[1]*0.5, self.sides[2]*0.5]
        ]
        self.neck_j = None
        self.eye_j = [None, None]
    
    def make_head(self, physics_client):
        """Create the head in the physics simulation"""
        # Create head body
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.sides[0]/2, self.sides[1]/2, self.sides[2]/2],
            rgbaColor=[0.5, 1.0, 0.5, 1],
            physicsClientId=physics_client
        )
        
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.sides[0]/2, self.sides[1]/2, self.sides[2]/2],
            physicsClientId=physics_client
        )
        
        self.body = p.createMultiBody(
            baseMass=1e-10,  # Dummy mass
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0.0, 0.0, 0.0],
            physicsClientId=physics_client
        )
        
        # Create eyes
        for i in range(2):
            eye_visual_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=self.radius,
                rgbaColor=[1.0, 0.5, 0.5, 1],
                physicsClientId=physics_client
            )
            
            eye_collision_id = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=self.radius,
                physicsClientId=physics_client
            )
            
            self.eye_b[i] = p.createMultiBody(
                baseMass=1e-10,  # Dummy mass
                baseCollisionShapeIndex=eye_collision_id,
                baseVisualShapeIndex=eye_visual_id,
                basePosition=self.pos[i],
                physicsClientId=physics_client
            )
            
            # Fix eyes to head
            self.eye_j[i] = p.createConstraint(
                parentBodyUniqueId=self.body,
                parentLinkIndex=-1,
                childBodyUniqueId=self.eye_b[i],
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=self.pos[i],
                childFramePosition=[0, 0, 0],
                physicsClientId=physics_client
            )

class RobotModel:
    """Complete robot model with body, head, and legs"""
    
    def __init__(self):
        self.geom = None
        self.body = None
        self.head = HeadModel()
        self.leg = [LegModel() for _ in range(NOL)]
        self.fixed = None
        self.sides = [1.0, 0.4, 0.2]  # length, width, height
        self.mass = 1.0
        
        # Leg positions relative to body
        self.px = (self.sides[0] - self.leg[0].sides[0]) * 0.5
        self.py = self.sides[1] * 0.5
        self.pz = self.sides[2] * 0.5
        
        self.legpos = [
            [self.px, self.py, self.pz],
            [0.0, self.py, self.pz],
            [-self.px, self.py, self.pz],
            [self.px, -self.py, self.pz],
            [0.0, -self.py, self.pz],
            [-self.px, -self.py, self.pz]
        ]
        
        # Rotation matrices
        self.Ri = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Identity
        self.Rz = [-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # 180° Z rotation
    
    def make_robot(self, physics_client):
        """Create the complete robot in the physics simulation"""
        # Create body
        pos = [0.0, 0.0, self.sides[2] * 0.5]
        
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.sides[0]/2, self.sides[1]/2, self.sides[2]/2],
            rgbaColor=[0.3, 0.7, 0.3, 1],
            physicsClientId=physics_client
        )
        
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.sides[0]/2, self.sides[1]/2, self.sides[2]/2],
            physicsClientId=physics_client
        )
        
        self.body = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=pos,
            physicsClientId=physics_client
        )
        
        # Create head
        self.head.make_head(physics_client)
        p.resetBasePositionAndOrientation(
            self.head.body,
            [pos[0] + self.sides[0] * 0.5, pos[1], pos[2] + self.sides[2] * 0.5],
            [0, 0, 0, 1],
            physicsClientId=physics_client
        )
        
        # Attach head to body
        self.head.neck_j = p.createConstraint(
            parentBodyUniqueId=self.body,
            parentLinkIndex=-1,
            childBodyUniqueId=self.head.body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[self.sides[0] * 0.5, 0, self.sides[2] * 0.5],
            childFramePosition=[0, 0, 0],
            physicsClientId=physics_client
        )
        
        # Create legs
        for i in range(NOL):
            self.leg[i].make_leg(physics_client)
            
            # Translate and rotate legs
            if i < 3:
                self.leg[i].translate(self.legpos[i], self.Ri, physics_client)  # Left legs
            else:
                self.leg[i].translate(self.legpos[i], self.Rz, physics_client)  # Right legs
            
            # Attach legs to body
            self.leg[i].joint[0] = p.createConstraint(
                parentBodyUniqueId=self.body,
                parentLinkIndex=-1,
                childBodyUniqueId=self.leg[i].body[0],
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=self.legpos[i],
                childFramePosition=[0, 0, 0],
                physicsClientId=physics_client
            )
        
        # Fix robot in air initially
        self.fixed = p.createConstraint(
            parentBodyUniqueId=self.body,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=physics_client
        )


def p_control(joint_id, target, physics_client):
    """Proportional control for joint movement"""
    kp = 5.0
    fmax = 20.0
    
    # Get current angle
    current_angle = p.getJointState(joint_id[0], joint_id[1], physicsClientId=physics_client)[0]
    diff = target - current_angle
    u = kp * diff
    
    p.setJointMotorControl2(
        bodyUniqueId=joint_id[0],
        jointIndex=joint_id[1],
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=u,
        force=fmax,
        physicsClientId=physics_client
    )


def move_robot(robot, tang, vel_counter, times, physics_client):
    """Move the robot according to target joint angles"""
    samstep = 20  # Sampling steps for feedback control
    
    vel_counter += 1
    
    # Apply controls to each joint
    for i in range(NOL):
        for j in range(NOJ):
            p_control([robot.leg[i].body[j+1], 0], tang[i][j], physics_client)
    
    # Update target joint angles periodically
    if vel_counter % samstep == 0:
        from genetic import loco_main
        loco_main(tang, times)
        vel_counter = 0
    
    return vel_counter


def release_robot(robot, physics_client):
    """Release the robot from fixed constraint"""
    p.removeConstraint(robot.fixed, physicsClientId=physics_client)