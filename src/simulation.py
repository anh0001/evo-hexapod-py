import pybullet as p
import pybullet_data
import time
import numpy as np
from robot import RobotModel, move_robot, release_robot
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
        
        # Create ground plane
        self.ground_id = p.createCollisionShape(
            shapeType=p.GEOM_PLANE,
            planeNormal=[0, 0, 1],
            physicsClientId=self.physics_client
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self.ground_id,
            basePosition=[0, 0, 0],
            physicsClientId=self.physics_client
        )
        
        # Create robot
        self.robot = RobotModel()
        try:
            self.robot.make_robot(self.physics_client)
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
        
        # Key bindings
        self.register_key_handlers()
    
    def register_key_handlers(self):
        """Set up keyboard event handlers"""
        # Register 's' key for starting simulation
        p.registerKeyCallback(ord('s'), self._on_start_key)
        
        # Register 'x' key for exiting
        p.registerKeyCallback(ord('x'), self._on_exit_key)
    
    def _on_start_key(self, key, is_down):
        """Handler for 's' key to start robot movement"""
        if is_down:
            self.action = True
            release_robot(self.robot, self.physics_client)
            self.environment.obstacles_free(self.physics_client)
            print("Robot released - simulation started")
    
    def _on_exit_key(self, key, is_down):
        """Handler for 'x' key to exit simulation"""
        if is_down:
            p.disconnect(self.physics_client)
            exit(0)
    
    def step_simulation(self):
        """Execute one step of the simulation"""
        if self.action:
            self.vel_counter, self.times = move_robot(
                self.robot, tang, self.vel_counter, 
                self.times, self.physics_client
            )
            
        # Step the simulation
        p.stepSimulation(self.physics_client)
    
    def run(self, max_steps=None):
        """Run the simulation loop"""
        steps = 0
        try:
            while max_steps is None or steps < max_steps:
                self.step_simulation()
                time.sleep(1/240)  # Limit to ~240 fps
                steps += 1
        except KeyboardInterrupt:
            pass
        finally:
            p.disconnect(self.physics_client)


def main():
    """Entry point for the simulation"""
    sim = Simulation()
    print("Simulation initialized.")
    print("Press 's' to start robot movement")
    print("Press 'x' to exit simulation")
    sim.run()


if __name__ == "__main__":
    main()