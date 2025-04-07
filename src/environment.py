import pybullet as p
import numpy as np
from genetic import rnd

# Constants
NOB = 5  # Number of obstacles per row/column

class Environment:
    """Environment with obstacles for robot simulation"""
    
    def __init__(self):
        self.c_body = [None] * (NOB * NOB)
        self.c_geom = [None] * (NOB * NOB)
        self.c_joint = [None] * (NOB * NOB)
        self.c_sides = [0.3, 0.3, 0.1]  # length, width, height
    
    def create_obstacles(self, physics_client):
        """Create obstacles in the environment"""
        box_mass = 1.0
        
        for i in range(NOB):
            for j in range(NOB):
                idx = i * NOB + j
                
                # Calculate position with random offset
                position = [(float(i) - 1.0 + rnd() * 0.5),
                           (float(j) - 2.0 + rnd() * 0.5),
                           self.c_sides[2] * 0.5]
                
                # Create visual shape
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[self.c_sides[0]/2, self.c_sides[1]/2, self.c_sides[2]/2],
                    rgbaColor=[1.0, 0.0, 1.0, 1.0],
                    physicsClientId=physics_client
                )
                
                # Create collision shape
                collision_shape_id = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[self.c_sides[0]/2, self.c_sides[1]/2, self.c_sides[2]/2],
                    physicsClientId=physics_client
                )
                
                # Create body
                self.c_body[idx] = p.createMultiBody(
                    baseMass=box_mass,
                    baseCollisionShapeIndex=collision_shape_id,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=position,
                    physicsClientId=physics_client
                )
                
                # Create fixed constraint to keep obstacle in place
                self.c_joint[idx] = p.createConstraint(
                    parentBodyUniqueId=self.c_body[idx],
                    parentLinkIndex=-1,
                    childBodyUniqueId=-1,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=position,
                    physicsClientId=physics_client
                )
    
    def obstacles_free(self, physics_client):
        """Remove constraints from obstacles to allow movement"""
        for i in range(NOB):
            for j in range(NOB):
                idx = i * NOB + j
                p.removeConstraint(self.c_joint[idx], physicsClientId=physics_client)


def create_ground(physics_client):
    """Create ground plane"""
    return p.createCollisionShape(
        shapeType=p.GEOM_PLANE,
        planeNormal=[0, 0, 1],
        physicsClientId=physics_client
    )