import numpy as np
from matplotlib import pyplot as plt
from IPython import embed as IPS
import matplotlib.animation as animation
import matplotlib.image as mpimg


class P_controller:
    def __init__(self, K_p, d_ref):
        self.K_p = K_p
        self.d_ref = d_ref

    def compute_error(self, my_position, forward_neighbor_position, backward_neighbor_position, errors_vehicle):
        if backward_neighbor_position is not None:
            self.error = (forward_neighbor_position - my_position) - (my_position - backward_neighbor_position)
            # self.error_sum = sum(errors_vehicle)
        else:
            self.error = (forward_neighbor_position - my_position) # -  forward error;
    def return_torque(self, my_position, forward_neighbor_position, backward_neighbor_position, errors_vehicle):
        self.compute_error(my_position, forward_neighbor_position, backward_neighbor_position, errors_vehicle)
        self.torque = self.K_p*self.error # + 100*self.error_sum
        return self.torque
    def return_error(self):
        return self.error

class vehicle:
    '''
    Initialize a vehicle with the following parameters:
    r ... wheel radius
    ga ... gravitational acceleration
    alpha ... road grade
    cr ... rolling resistance coefficient
    rho ... air density
    Ar ... cross-sectional area of the vehicle
    Cd ... aerodynamic drag coefficient
    m ... mass
    dt ... discrete time step
    dummy_param ... some dummy parameter that does not influence the model
    Of these parameters we assume that we have knowledge of m, alpha, and some dummy parameter
    '''
    def __init__(self,r,ga,alpha,cr,rho,Ar,Cd,m,dt, s_init):
        self.r = r
        self.ga = ga
        self.alpha = alpha
        self.cr = cr
        self.rho = rho
        self.Ar = Ar
        self.Cd = Cd
        self.m = m
        self.dt = dt
        self.v = 0
        self.s = s_init
        self.x = np.vstack((self.s, self.v))
        self.max_speed = np.infty
        self.max_torque = 50000
        self.u = 0

    def dynamics(self,u):
        # u = np.clip(u, -self.max_torque, self.max_torque)  # Limit torque within valid range; anyways input
        self.s = self.s + self.v*self.dt
        self.v = min(self.v + ((self.Ft(u) + self.Fg() + self.Fr() + self.Fd())/self.m)*self.dt + np.random.normal(0,1e-4,1), self.max_speed)  # Limit velocity within limit 
        self.x = np.vstack((self.s,self.v))

    # Gravitational Force
    def Fg(self):
        return -self.m*self.ga*np.sin(self.alpha)

    # Rolling Resistance
    def Fr(self):
        if self.v > 0:
            return -self.cr*self.m*self.ga 
        elif self.v < 0:
            return self.cr*self.m*self.ga 
        else:
            return 0

    # Aerodynamic Drag
    def Fd(self):
        return -0.5*self.rho*self.Ar*self.Cd*self.v**2

    # Forward Torque
    def Ft(self,u):
        return (1/self.r)*u

class leading_vehicle(vehicle):
    def __init__(self,r,ga,alpha,cr,rho,Ar,Cd,m, dt, s_init, v):
        super().__init__(r,ga,alpha,cr,rho,Ar,Cd,m,dt,s_init)
        self.v = v
    def dynamics(self):
        # u = np.clip(u, -self.max_torque, self.max_torque)  # Limit torque within valid range; anyways input
        self.s = self.s + self.v*self.dt
        self.x = np.vstack((self.s,self.v))


def simulate(hyperparameters, K_p_values):
    num_vehicles, v_leader, d_ref, steps, dt = hyperparameters
    list_vehicles = []
    list_controllers = []
    for i in range(num_vehicles):
        r = np.random.uniform(0.1, 0.8) # Wheel radius (m)
        ga = 9.81      # Gravitational acceleration (m/s²); all on same planet
        alpha = 0.05   # Road grade (rad) ≈ 2.86°; all on same street
        cr = np.random.uniform(0.004, 0.008)  # 0.006     # Rolling resistance coefficient
        rho = 1.225    # Air density (kg/m³); all in the same air
        Ar = np.random.uniform(8, 12)  # 10        # Cross-sectional area (m²)
        Cd = np.random.uniform(0.4, 0.8)  # 0.6       # Aerodynamic drag coefficient
        m = np.random.uniform(8000, 16000)  # 15000      # Mass (kg) ~15 tons
        if i != num_vehicles - 1:  # not the leading vehicle with const. velocity dynamics
            list_vehicles.append(vehicle(r, ga, alpha, cr, rho, Ar, Cd, m, dt, s_init=(10-i)*20))
            list_controllers.append(P_controller(K_p=K_p_values[i], d_ref=d_ref))
        else:
            list_vehicles.append(leading_vehicle(r, ga, alpha, cr, rho, Ar, Cd, m, dt, s_init=250, v=v_leader))

    # Simulation settings
     # Number of simulation steps
    
    initial_torque = 0

    # Set up controller for each agent
    positions = np.zeros([steps, num_vehicles])
    velocities = np.zeros([steps, num_vehicles])
    errors = np.zeros([steps, num_vehicles-1])
    torques = np.zeros([steps, num_vehicles-1])

    torque = initial_torque
    for i in range(steps):
        for j in range(num_vehicles):
            positions[i, j] = list_vehicles[j].s
            velocities[i, j] = list_vehicles[j].v
        for j in range(num_vehicles):
            if j != num_vehicles - 1:  # not the leading vehicle
                my_position = positions[i, j] 
                forward_neighbor_position = positions[i, j+1]
                if j != 0:
                    backward_neighbor_position = positions[i, j-1]
                    torque = list_controllers[j].return_torque(my_position=my_position, forward_neighbor_position=forward_neighbor_position,
                     backward_neighbor_position=backward_neighbor_position, errors_vehicle=errors[:, j])
                if j == 0:
                    torque = list_controllers[j].return_torque(my_position=my_position, forward_neighbor_position=forward_neighbor_position,
                     backward_neighbor_position=None, errors_vehicle=errors[:, j])
                list_vehicles[j].dynamics(torque)  # Apply torque
                errors[i, j]=list_controllers[j].error
                torques[i, j]=torque
            else:  # if leading vehicle
                list_vehicles[j].dynamics()
    return positions, velocities, errors, torques

def reward_function(errors, positions, d_ref, num_vehicles):
    sum_errors = sum([sum(abs(errors[:, j])) for j in range(num_vehicles-1)])
    sorted_positions = np.sort(positions, axis=1)
    # Compute differences between consecutive sorted elements
    diffs = np.diff(sorted_positions, axis=1)  # don't do this; we have a sorting and if ever i-1 gets in front of i, we have a collision.
    # Take the minimum difference per row
    min_diff = min(np.min(diffs, axis=1))


    collision_penalty = num_vehicles*(min_diff - d_ref)
    return - sum_errors - collision_penalty  # this is not properly scaled yet
if __name__ == '__main__':
    num_vehicles = 3
    v_leader = 10
    T = 50  # Total simulation time in seconds
    dt = 0.1
    steps = int(T / dt) 
    # Determine goal distance
    d_ref = 10  # we want 10m between the LKWs
    time = np.linspace(0, T, steps)
    hyperparameters_simulation = [num_vehicles, v_leader, d_ref, steps, dt]
    K_p_values = [400]*num_vehicles
    positions, velocities, errors, torques  = simulate(hyperparameters_simulation, K_p_values)  # we will tune K_p values from our algoritm
    reward = reward_function(errors, positions, d_ref, num_vehicles)
    
    
    # Plot results
    plt.figure()
    for j in range(num_vehicles):
        plt.plot(time, positions[:, j], label=f"LKW {j}")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    for j in range(num_vehicles):
        plt.plot(time, velocities[:, j], label=f'{j}')
    plt.xlabel("Time (s)")
    plt.legend()
    plt.ylabel("Velocity (m/s)")
    plt.grid()
    plt.show()

    plt.figure()
    for j in range(num_vehicles-1):
        plt.plot(time, torques[:, j], label=f'{j}')
    plt.xlabel("Time")
    plt.legend()
    plt.ylabel("Torque")

    plt.figure()
    for j in range(num_vehicles-1):
        plt.plot(time, errors[:, j], label=f'{j}')
    plt.xlabel("Time")
    plt.legend()
    plt.ylabel("Error")



    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["b", "r", "g", "m", "k"]  # Blue, Red, Green, Magenta
    ax.set_xlim(np.min(positions)-10, np.max(positions) + 10)  # Extend the x-axis dynamically
    ax.set_ylim(-0.2, 0.2)  # Keep the y-axis fixed
    ax.set_xlabel("Position (m)")
    ax.set_title("LKW Movement Simulation")

    # Draw the road
    road, = ax.plot([np.min(positions)-10, np.max(positions) + 10], [0, 0], "k-", linewidth=2)

    vehicle_img = mpimg.imread("truck.png")  # PNG with transparent background recommended
    # x = 200  # Arbitrary position along the x-axis
    # # ax.set_ylim(-10, 10)
    # test_marker = ax.imshow(vehicle_img, extent=(x - 50, x + 50, -50, 50), zorder=10)
    # plt.show()
    # Initialize markers for all LKWs with different colors
    LKW_markers = [
        ax.plot([], [], f"{colors[j]}o", markersize=20, label=f"LKW {j+1}")[0] 
        for j in range(num_vehicles)
    ]
    # LKW_markers = [
    #     ax.imshow(vehicle_img, extent=(0, 2, -0.5, 0.5), zorder=10)  # Initial dummy extent
    #     for _ in range(num_vehicles)
    # ]


    # Update function for animation
    def update(frame):
        for j in range(num_vehicles):
            LKW_markers[j].set_data(positions[frame, j], 0)  # Move each LKW along the x-axis
            # x = positions[frame, j]
            # LKW_markers[j].set_extent((x - 100, x + 100, -3, 3))
        return LKW_markers

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
    # ani.save("lkw_simulation.gif", writer="pillow", fps=20)
    # plt.legend()
    plt.show()
    print("Done")







    # plt.legend()
    # plt.show()



    