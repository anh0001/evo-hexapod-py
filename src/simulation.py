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
        
        # Set gravity (matching original -9.8 m/s^2)
        p.setGravity(0, 0, -9.8)
        
        # Set simulator parameters (similar to ODE ERP and CFM)
        p.setPhysicsEngineParameter(
            fixedTimeStep=0.02,  # 50Hz simulation
            numSolverIterations=50,
            numSubSteps=2,
            erp=0.2,  # Error reduction parameter (like in ODE)
            contactERP=0.2,
            frictionERP=0.2,
            globalCFM=1.0e-10  # Constraint force mixing (like in ODE)
        )
        
        # Set up visualization
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        
        # Set initial camera position (matching original)
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=101.0,
            cameraPitch=-27.5,
            cameraTargetPosition=[0, 0, 0.5]
        )
        
        # Create ground plane with friction parameters matching original
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
        
        # Set ground friction parameters similar to the original ODE model
        p.changeDynamics(
            ground_body, 
            -1, 
            lateralFriction=1.0,
            restitution=0.2,  # BOUNCE parameter from original
            contactDamping=0.8,  # ERP parameter
            contactStiffness=1/0.00001,  # Inverse of CFM
            physicsClientId=self.physics_client
        )
        
        # Create robot
        try:
            success = robot.make_robot(self.physics_client)
            if success:
                print("Robot created successfully")
            else:
                print("Failed to create robot")
                raise RuntimeError("Robot creation failed")
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
        self.should_exit = False
        
        # Register keyboard callback
        p.setRealTimeSimulation(0)  # We'll step manually
        
        # Print instructions
        print("\n==== Hexapod Evolution Simulation ====")
        print("Press 's' to start robot movement")
        print("Press 'x' to exit simulation")
        print("=====================================\n")
    
    def process_keyboard_events(self):
        """Process keyboard events"""
        keys = p.getKeyboardEvents()
        
        # Check for 's' key press (ASCII 115)
        if 115 in keys and keys[115] & p.KEY_WAS_TRIGGERED:
            self.action = True
            release_robot(self.physics_client)
            self.environment.obstacles_free(self.physics_client)
            print("Robot released - evolution started")
        
        # Check for 'x' key press (ASCII 120)
        if 120 in keys and keys[120] & p.KEY_WAS_TRIGGERED:
            print("Exiting simulation")
            self.should_exit = True
    
    def step_simulation(self):
        """Execute one step of the simulation"""
        # Process keyboard events
        self.process_keyboard_events()
        
        # Check if we should exit
        if self.should_exit:
            return False
        
        # Update robot control if action is enabled
        if self.action:
            self.vel_counter, self.times = move_robot(
                tang, self.vel_counter, 
                self.times, self.physics_client
            )
        
        # Step collision detection and physics (matches original ODE near callback)
        p.stepSimulation(self.physics_client)
        
        return True  # Continue simulation
    
    def run(self, max_steps=None):
        """Run the simulation loop"""
        steps = 0
        time_step = 1/50.0  # 50 Hz to match original
        
        try:
            while (max_steps is None or steps < max_steps) and not self.should_exit:
                start_time = time.time()
                
                if not self.step_simulation():
                    break  # Exit if step_simulation returns False
                
                # Calculate time to sleep to maintain consistent simulation rate
                elapsed = time.time() - start_time
                sleep_time = max(0, time_step - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
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