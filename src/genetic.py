import numpy as np
import math
import pybullet as p

# Constants
NOL = 6  # Number of legs
NOJ = 3  # Number of joints per leg
DOF = 3  # Degree of freedom (evaluation behaviors)
GAN = 30  # Population size
GAL = 10  # Maximum gene length (max number of postures)
TRL = 500  # Maximum number of iterations

# Global variables
thostl = [0] * GAN  # Gene length (number of postures) for each individual
gac = [-1] * GAN  # Group number
gai = 0  # Individual ID for simulation
gaj = 0  # Posture sequence ID

# View parameters
hpr = [101.0, -27.5, 0.0]  # View direction
xyz2 = [0.8317, -2.9817, 2.0]  # Viewpoint position
posz = 1  # normal:1; overturn:-1 flag

# Robot position and orientation
rp = [0.0, 0.0, 0.0]  # Robot position
rpp = [0.0, 0.0, 0.0]  # Previous robot position
ra = 0.0  # Robot moving angle
rap = 0.0  # Previous robot moving angle
rr = [1.0, 0.0, 0.0]  # Robot posture
rrp = [0.0, 0.0, 0.0]  # Previous robot posture

# Counters
iteration = 0  # Iteration counter (generation number)
timesmax = 20  # Max posture change count
times = 0  # Posture change counter

# Joint angle limits and initial values
qmin = [-45.0, 0.0, 0.0]  # Min angle for target motion (deg)
qrange = [90.0, 60.0, 60.0]  # Range for target motion (deg)
qinit = [0.0, 45.0, 45.0]  # Init angle for target motion (deg)
tang = [[0.0 for _ in range(NOJ)] for _ in range(NOL)]  # Target joint angle (rad)

# Genetic algorithm data structures
thost = [[[[0.0 for _ in range(NOJ)] for _ in range(2)] for _ in range(GAL)] for _ in range(GAN)]  # Gene sequences
bfith = [[0.0 for _ in range(DOF)] for _ in range(TRL)]  # Best fitness for each generation
fith = [[0.0 for _ in range(DOF+1)] for _ in range(GAN)]  # Fitness values for current population


def rnd():
    """Generate uniform random number between 0 and 1"""
    return np.random.random()


def rndn():
    """Generate normal random number"""
    return sum(rnd() for _ in range(12)) - 6.0


def rad(angle):
    """Convert degrees to radians"""
    return angle * math.pi / 180.0


def deg(angle):
    """Convert radians to degrees"""
    return angle * 180.0 / math.pi


def vega_rank():
    """Rank individuals by fitness for each evaluation criterion"""
    global gac
    
    # Reset group assignments
    for i in range(GAN):
        gac[i] = -1
    
    # Assign individuals to groups based on fitness
    for j in range(GAN):
        h = j % DOF  # h = (0, 1, 2)
        k = 0
        
        # Find first unassigned individual
        while k < GAN and gac[k] != -1:
            k += 1
        
        if k >= GAN:
            break  # All individuals already assigned
        
        # Find individual with highest fitness for criterion h
        for i in range(k+1, GAN):
            if gac[i] == -1 and fith[i][h] > fith[k][h]:
                k = i
        
        # Assign individual to group
        gac[k] = h


def vega_main():
    """Vector Evaluated Genetic Algorithm main function"""
    global gai, thostl
    
    # Behavior names for logging
    tn = ["Forward", "Left Turn", "Right Turn"]
    
    # Rank individuals by fitness
    vega_rank()
    
    # Select criterion for this generation
    h = iteration % DOF
    
    # Find individuals in the current group
    g1 = 0
    while g1 < GAN and gac[g1] != h:
        g1 += 1
    
    if g1 >= GAN:
        print(f"No individuals found for group {h}")
        g1 = 0
        gac[g1] = h
    
    g2 = g1
    
    # Find worst and best individuals in group
    for i in range(g1+1, GAN):
        if gac[i] == h:
            if fith[i][h] < fith[g1][h]:
                g1 = i  # g1 is worst individual
            elif fith[i][h] > fith[g2][h]:
                g2 = i  # g2 is best individual
    
    if iteration < 100:
        print(f"Search for {tn[h]}")
        
        # Step 1: Reproduction + elite crossover + simple mutation
        g3 = int(GAN * rnd())  # Random individual
        r = rnd() * 0.5
        
        # Copy gene length from best individual
        thostl[g1] = thostl[g2]
        
        # Genetic operations
        for m in range(thostl[g1]):
            for i in range(2):
                for j in range(NOJ):
                    if rnd() < r and m < thostl[g3]:
                        # Copy from random individual with mutation
                        thost[g1][m][i][j] = thost[g3][m][i][j] + rndn() * qrange[j] * 0.2
                    else:
                        # Copy from best individual with mutation
                        thost[g1][m][i][j] = thost[g2][m][i][j] + rndn() * qrange[j] * 0.1
                    
                    # Ensure angle is within limits
                    if thost[g1][m][i][j] < qmin[j]:
                        thost[g1][m][i][j] = qmin[j] + rnd() * 0.01
                    elif thost[g1][m][i][j] > qmin[j] + qrange[j]:
                        thost[g1][m][i][j] = qmin[j] + qrange[j] - rnd() * 0.01
        
        # Step 2: Insertion or deletion mutation
        if thostl[g1] < GAL - 1 and rnd() < 0.15:
            # Insertion mutation - add a posture
            print("-- insertion mutation  --")
            k = int(thostl[g1] * rnd())
            
            if k < thostl[g1]:
                # Shift postures to make room
                for m in range(thostl[g1], k, -1):
                    for i in range(2):
                        for j in range(NOJ):
                            thost[g1][m][i][j] = thost[g1][m-1][i][j]
                
                # Insert new random posture
                for i in range(2):
                    for j in range(NOJ):
                        thost[g1][k][i][j] = qmin[j] + qrange[j] * rnd()
            
            thostl[g1] += 1
            
        elif thostl[g1] > 2 and rnd() < 0.15:
            # Deletion mutation - remove a posture
            thostl[g1] -= 1
            print("-- deletion mutation  --")
            k = int(thostl[g1] * rnd())
            
            if k < thostl[g1] - 1:
                # Shift remaining postures
                for m in range(k, thostl[g1]):
                    for i in range(2):
                        for j in range(NOJ):
                            thost[g1][m][i][j] = thost[g1][m+1][i][j]
        
        # Step 3: Exchange mutations
        if rnd() < 0.1:
            # Phase exchange - swap left/right leg postures
            print("-- phase exchange mutation  --")
            m = int(thostl[g1] * rnd())
            for j in range(NOJ):
                d = thost[g1][m][0][j]
                thost[g1][m][0][j] = thost[g1][m][1][j]
                thost[g1][m][1][j] = d
        
        elif rnd() < 0.1:
            # Order exchange - swap postures in sequence
            k = int(thostl[g1] * rnd())
            m = int(thostl[g1] * rnd())
            
            if k != m:
                print("-- order exchange mutation  --")
                for i in range(2):
                    for j in range(NOJ):
                        d = thost[g1][m][i][j]
                        thost[g1][m][i][j] = thost[g1][k][i][j]
                        thost[g1][k][i][j] = d
        
        # Use modified individual for next simulation
        gai = g1
    
    else:
        # After 100 generations, use best individual
        print(f"Best Locomotion of {tn[h]}")
        gai = g2


def loco_main(tang, times):
    """Update locomotion based on genetic data"""
    global gaj, posz, iteration, gai  # Added gai to globals
    global rp, rpp, ra, rap, rr, rrp
    
    # Increment counters
    times += 1
    gaj += 1
    
    # Import robot here to avoid circular imports
    from robot import robot, get_base_position_and_orientation, get_base_rotation
    
    if times > timesmax:
        # Evaluation phase - calculate fitness for current individual
        rap = ra
        for i in range(3):
            rpp[i] = rp[i]
            rrp[i] = rr[i]
        
        # Get current position and orientation
        pos0, _ = get_base_position_and_orientation(p)
        rot0 = get_base_rotation(p)
        
        # Calculate rotation angle
        if rot0[4] == 0 and rot0[0] == 0:
            ra = 0
        else:
            ra = math.atan2(rot0[4], rot0[0])
        
        a = ra - rap
        if a > math.pi:
            a -= 2 * math.pi
        elif a < -math.pi:
            a += 2 * math.pi
        
        # Calculate fitness components with more robust formulas
        f = [0.0] * 5
        f[0] = math.exp(-a * a)  # Go straight - higher when angle change is small
        f[1] = math.exp(-(a + math.pi * 0.5) ** 2)  # Turn left - higher when turning left
        f[2] = math.exp(-(a - math.pi * 0.5) ** 2)  # Turn right - higher when turning right
        
        # Save robot orientation
        rr[0] = rot0[0]
        rr[1] = rot0[4]
        
        # Calculate movement distance and direction
        d = 0.0
        q = 0.0
        v = [0.0] * 5
        
        for i in range(2):
            rp[i] = pos0[i]
            v[i] = rp[i] - rpp[i]
            d += v[i] * v[i]
            q += rr[i] * v[i]  # Inner product between orientation and movement
        
        d = math.sqrt(d)  # Movement distance
        if d > 0.001:  # Avoid division by zero with threshold
            q = q / d  # Cosine of angle between movement and orientation
        else:
            q = 0.0  # No movement, so no direction
        
        # Calculate fitness for each objective with balanced weights
        fith[gai][0] = f[0] + d * 10 + q  # Forward movement: straight + distance + alignment
        fith[gai][1] = f[1] * 20 + math.exp(-d * d)  # Left turn: turning + minimal distance
        fith[gai][2] = f[2] * 20 + math.exp(-d * d)  # Right turn: turning + minimal distance
        
        # Calculate total fitness as sum of individual objectives
        fith[gai][3] = sum(fith[gai][i] for i in range(DOF))
        
        # Check if robot is upside down
        if rot0[10] < -0.7:
            posz = -1
        else:
            posz = 1
        
        # Print status
        print(f"[{gai}] walking distance: {d:.3f}, posture change: {a:.3f}, moving dir: {q:.3f}, "
              f"({rot0[0]:.2f}, {rot0[1]:.2f}, {rot0[2]:.2f})")
        print(f"Current fit[0,F]: {fith[gai][0]:.3f}, fit[1,L]: {fith[gai][1]:.3f}, "
              f"fit[2,R]: {fith[gai][2]:.3f}, pos-z: ({rot0[10]:.2f})")
        
        # Record best fitness for this generation
        h = min(iteration + 1, GAN)
        for j in range(DOF):
            k = 0
            for i in range(h):
                if fith[i][j] > fith[k][j]:
                    k = i
            bfith[iteration][j] = fith[k][j]
        
        # Prepare for next generation
        iteration += 1
        
        # Update camera position
        for i in range(2):
            xyz2[i] += rp[i] - rpp[i]
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=xyz2[0],
            cameraPitch=xyz2[1],
            cameraTargetPosition=[rp[0], rp[1], rp[2]]
        )
        
        # Select next individual
        if iteration < GAN:
            gai = iteration
        else:
            vega_main()
        
        # Reset to initial posture
        for i in range(NOL):
            for j in range(NOJ):
                tang[i][j] = rad(qinit[j])
        
        # Reset counters
        gaj = -1
        print(f"Iterations: {iteration}, host: {gai}")
        times = 0
        
    else:
        # Normal movement - update target angles based on current posture sequence
        if thostl[gai] > 0:  # Avoid division by zero
            gaj = gaj % thostl[gai]
            
            for j in range(NOJ):
                for i in range(NOL):
                    if j == 0:  # Shoulder
                        if i % 2 == 0:  # Legs 0, 2, 4
                            if i < 3:
                                tang[i][j] = rad(thost[gai][gaj][0][j])  # Legs 0, 2
                            else:
                                tang[i][j] = -rad(thost[gai][gaj][0][j])  # Leg 4
                        else:  # Legs 1, 3, 5
                            if i < 3:
                                tang[i][j] = rad(thost[gai][gaj][1][j])  # Leg 1
                            else:
                                tang[i][j] = -rad(thost[gai][gaj][1][j])  # Legs 3, 5
                    else:  # Elbow
                        if i % 2 == 0:
                            tang[i][j] = rad(thost[gai][gaj][0][j])  # Legs 0, 2, 4
                        else:
                            tang[i][j] = rad(thost[gai][gaj][1][j])  # Legs 1, 3, 5
    
    # Adjust for flipped orientation
    for i in range(NOL):
        for j in range(NOJ):
            tang[i][j] *= posz
    
    return times


def robot_init():
    """Initialize robot population with random posture sequences"""
    global thostl, thost, tang
    
    # Initialize arrays
    for i in range(TRL):
        for j in range(DOF):
            bfith[i][j] = 0.0
    
    # Set initial target posture
    for i in range(NOL):
        for j in range(NOJ):
            tang[i][j] = rad(qinit[j])
    
    # Initialize population
    for n in range(GAN):
        for j in range(DOF):
            fith[n][j] = 0.0
        
        # Random gene length (number of postures) for each individual
        thostl[n] = 2 + int(rnd() * 3)
        
        for m in range(thostl[n]):
            print(f"Host [{n}][{m}]")
            for i in range(2):  # Left/right phases
                for j in range(NOJ):
                    # Random initial joint angles
                    thost[n][m][i][j] = qmin[j] + qrange[j] * rnd()
                    print(f"thost[{n}][{m}][{i}][{j}]: {thost[n][m][i][j]}")