import pybullet as p
import pybullet_data
import time
import numpy as np
import os

from robot import robot, move_robot, release_robot
from environment import Environment, create_ground
from genetic import robot_init, tang, timesmax

class Simulation:
    """Main simulation class for evolutionary robotics"""
    
    def __init__(self):
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set gravity
        p.setGravity(0, 0, -9.8)
        
        # Set simulator parameters (similar to ODE ERP and CFM)
        p.setPhysicsEngineParameter(
            fixedTimeStep=0.02,
            numSolverIterations=50,
            numSubSteps=2,
            erp=0.2,  # Error reduction parameter
            contactERP=0.2,
            frictionERP=0.2,
            globalCFM=1.0e-10  # Constraint force mixing
        )
        
        # Set up visualization
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        
        # Set camera position
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=135.0,
            cameraPitch=-10.0,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # Create ground plane with friction
        self.ground_id = p.createCollisionShape(
            shapeType=p.GEOM_PLANE,
            planeNormal=[0, 0, 1],
            physicsClientId=self.physics_client
        )
        
        ground_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self.ground_id,
            basePosition=[0, 0, 0],
            physicsClientId=self.physics_client
        )
        
        # Set ground friction
        p.changeDynamics(
            ground_body, 
            -1, 
            lateralFriction=0.8,
            spinningFriction=0.1,
            rollingFriction=0.1,
            physicsClientId=self.physics_client
        )
        
        # Create robot
        try:
            robot.make_robot(self.physics_client)
            print("Robot created successfully")
        except Exception as e:
            print(f"Error creating robot: {e}")
            raise
        
        # Create environment with obstacles
        self.environment = Environment()
        self.environment.create_obstacles(self.physics_client)
        
        # Initialize genetic algorithm
        robot_init()
        
        # Simulation variables
        self.vel_counter = 0
        self.times = 0
        self.action = False
        self.should_exit = False  # Flag to indicate when to exit
        
        # Print instructions
        print("Press 's' to start robot movement")
        print("Press 'x' to exit simulation")
    
    def process_keyboard_events(self):
        """Process keyboard events"""
        keys = p.getKeyboardEvents()
        
        # Check for 's' key press (ASCII 115)
        if 115 in keys and keys[115] & p.KEY_WAS_TRIGGERED:
            self.action = True
            release_robot(self.physics_client)
            self.environment.obstacles_free(self.physics_client)
            print("Robot released - simulation started")
        
        # Check for 'x' key press (ASCII 120)
        if 120 in keys and keys[120] & p.KEY_WAS_TRIGGERED:
            print("Exiting simulation")
            self.should_exit = True  # Set flag to exit instead of calling exit(0)
    
    def step_simulation(self):
        """Execute one step of the simulation"""
        # Process keyboard events
        self.process_keyboard_events()
        
        # Check if we should exit
        if self.should_exit:
            return False
        
        if self.action:
            self.vel_counter, self.times = move_robot(
                tang, self.vel_counter, 
                self.times, self.physics_client
            )
            
        # Step the simulation
        p.stepSimulation(self.physics_client)
        return True  # Continue simulation
    
    def run(self, max_steps=None):
        """Run the simulation loop"""
        steps = 0
        try:
            while (max_steps is None or steps < max_steps) and not self.should_exit:
                if not self.step_simulation():
                    break  # Exit if step_simulation returns False
                time.sleep(1/240)  # Limit to ~240 fps
                steps += 1
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        finally:
            try:
                if p.isConnected(self.physics_client):
                    p.disconnect(self.physics_client)
                    print("Disconnected from physics server")
            except:
                # Ignore errors during disconnect
                pass


def main():
    """Entry point for the simulation"""
    sim = Simulation()
    print("Simulation initialized.")
    sim.run()
    print("Simulation terminated.")


if __name__ == "__main__":
    main()