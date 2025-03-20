import numpy as np
from matplotlib import pyplot as plt
from IPython import embed as IPS
import matplotlib.animation as animation


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
    def __init__(self,r,ga,alpha,cr,rho,Ar,Cd,m,dt):
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
        self.s = 0
        self.x = np.vstack((self.s, self.v))
        self.max_speed = 22
        self.max_torque = 50000
        self.u = 0

    def dynamics(self,u):
        u = np.clip(u, -self.max_torque, self.max_torque)  # Limit torque within valid range; anyways input
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


if __name__ == '__main__':
    r = 0.5        # Wheel radius (m)
    ga = 9.81      # Gravitational acceleration (m/s²)
    alpha = 0.05   # Road grade (rad) ≈ 2.86°
    cr = 0.006     # Rolling resistance coefficient
    rho = 1.225    # Air density (kg/m³)
    Ar = 10        # Cross-sectional area (m²)
    Cd = 0.6       # Aerodynamic drag coefficient
    m = 15000      # Mass (kg) ~15 tons
    dt = 0.1       # Time step (s)

    LKW_1 = vehicle(r, ga, alpha, cr, rho, Ar, Cd, m, dt)
    LKW_2 = vehicle(r, ga, alpha, cr, rho, Ar, Cd, m, dt)
    LKW_3 = vehicle(r, ga, alpha, cr, rho, Ar, Cd, m, dt)
    LKW_4 = vehicle(r, ga, alpha, cr, rho, Ar, Cd, m, dt)
    LKWs = [LKW_1, LKW_2, LKW_3, LKW_4]
    T = 50  # Total simulation time in seconds
    steps = int(T / dt)  # Number of simulation steps
    u_1 = 40e3  # Constant torque applied (Nm)
    u_2 = 30e3  # Constant torque applied (Nm)
    u_3 = 20e3  # Constant torque applied (Nm)
    u_4 = 10e3  # Constant torque applied (Nm)
    torques = [u_1, u_2, u_3, u_4]

    time = np.linspace(0, T, steps)
    positions = np.zeros([steps, len(LKWs)])
    velocities = np.zeros([steps, len(LKWs)])


    for i in range(steps):
        for j in range(len(LKWs)):
            LKWs[j].dynamics(torques[j])  # Apply torque
            positions[i, j] = LKWs[j].s
            velocities[i, j] = LKWs[j].v

    # Plot results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for j in range(len(LKWs)):
        plt.plot(time, positions[:, j], label=f"LKW {j}")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    for j in range(len(LKWs)):
        plt.plot(time, velocities[:, j], color="r")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()

    plt.show()


    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["b", "r", "g", "m"]  # Blue, Red, Green, Magenta
    ax.set_xlim(0, np.max(positions) + 10)  # Extend the x-axis dynamically
    ax.set_ylim(-1, 1)  # Keep the y-axis fixed
    ax.set_xlabel("Position (m)")
    ax.set_title("LKW Movement Simulation")

    # Draw the road
    road, = ax.plot([0, np.max(positions) + 10], [0, 0], "k-", linewidth=2)

    # Initialize markers for all LKWs with different colors
    LKW_markers = [
        ax.plot([], [], f"{colors[j]}o", markersize=10, label=f"LKW {j+1}")[0] 
        for j in range(len(LKWs))
    ]

    # Update function for animation
    def update(frame):
        for j in range(len(LKWs)):
            LKW_markers[j].set_data(positions[frame, j], 0)  # Move each LKW along the x-axis
        return LKW_markers

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=True)

    plt.legend()
    plt.show()



    