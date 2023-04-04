#Author: Reza Samarghandi
#
#Email: RezaSamarghandi@yahoo.com

import math
import numpy as np
import matplotlib.pyplot as plt



G = 6.67428e-11 # Gravitational Constant
AU = (149.6e6 * 1000)
SCALE = 30 / AU # Defining Scale

fig = plt.figure()
aaa = fig.add_subplot(111, projection='3d')
# draw and show it
fig = plt.gcf()
fig.show()
fig.canvas.draw()

class Body():#Turtle):
    # Defining the Body which we want to calculate it's acceleration
    name = 'Body'
    mass = None  # Mass
    vx = vy = vz = 0.0  # Velocity
    px = py = pz = 0.0  # Position
    j2 = j3 = j4 = j5 = j6 = re = 0.0 

    def attraction(self, other):
        sx, sy, sz = self.px, self.py, self.pz
        ox, oy, oz = other.px, other.py, other.pz
        j2, j3, j4, j5, j6, re = other.j2, other.j3, other.j4, other.j5, other.j6, other.re
        mu = G * other.mass

        dx = (sx - ox)
        dy = (sy - oy)
        dz = (sz - oz)
        d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        ps = np.array([dx, dy, dz])

        aj2 = -3 / 2 * j2 * (mu / d ** 2) * (re / d) ** 2 * np.array(
            [((1 - 5 * (dz / d) ** 2) * (dx / d)), ((1 - 5 * (dz / d) ** 2) * (dy / d)),
             ((3 - 5 * (dz / d) ** 2) * (dz / d))])
        aj3 = 1 / 2 * j3 * (mu / d ** 2) * (re / d) ** 3 * np.array(
            [(5 * (7 * (dz / d) ** 3) * (dx / d)), (5 * (7 * (dz / d) ** 3 - 3 * (dz / d)) * (dy / d)),
             (3 * (1 - 10 * (dz / d) ** 2 + 35 / 3 * (dz / d) ** 4))])
        aj4 = 5 / 8 * j4 * mu / d ** 2 * (re / d) ** 4 * np.array(
            [((3 - 42 * (dz / d) ** 2 + 63 * (dz / d) ** 4) * (dx / d)),
             ((3 - 42 * (dz / d) ** 2 + 63 * (dz / d) ** 4) * (dy / d)),
             ((15 - 70 * (dz / d) ** 2 + 63 * (dz / d) ** 4) * (dz / d))])
        aj5 = j5 / 8 * mu / (d ** 2) * (re / d) ** 5 * np.array(
            [(3 * (35 * (dz / d) - 210 * (dz / d) ** 3 + 231 * (dz / d) ** 5) * (dx / d)),
             (3 * (35 * (dz / d) - 210 * (dz / d) ** 3 + 231 * (dz / d) ** 5) * (dy / d)),
             (693 * (dz / d) ** 6 - 945 * (dz / d) ** 4 + 315 * (dz / d) ** 2 - 15)])
        aj6 = -j6 / 16 * mu / (d ** 2) * (re / d) ** 6 * np.array(
            [((35 - 945 * (dz / d) ** 2 + 3465 * (dz / d) ** 4 - 3003 * (dz / d) ** 6) * (dx / d)),
             ((35 - 945 * (dz / d) ** 2 + 3465 * (dz / d) ** 4 - 3003 * (dz / d) ** 6) * (dy / d)),
             ((245 - 2205 * (dz / d) ** 2 + 4851 * (dz / d) ** 4 - 3003 * (dz / d) ** 6) * (dz / d))])

        aa = -mu * ps / d ** 3 + aj2 + aj3 + aj4 + aj5 + aj6 # Gravitational Perturbed acceleration 
        ax = aa[0]
        ay = aa[1]
        az = aa[2]
        return ax, ay, az  


def update_info(step, bodies):
    print('Step #{}'.format(step))
    for body in bodies:
        s = '{:<8}  Pos.={:>6.2f} {:>6.2f} {:>6.2f} Vel.={:>10.3f} {:>10.3f} {:>10.3f}'.format(
            body.name, body.px / AU, body.py / AU, body.pz / AU, body.vx, body.vy, body.vz)
        print(s)
    print()


def loop(bodies):
    timestep = 24 * 3600 # Timestep of simulation


    step = 1
    while True:
        update_info(step, bodies)
        step += 1

        Acceleration = {}
        for body in bodies:
            total_ax = total_ay = total_az = 0.0
            for other in bodies:
                if body is other:
                    continue
                ax, ay, az = body.attraction(other)
                total_ax += ax
                total_ay += ay
                total_az += az
            Acceleration[body] = (total_ax, total_ay, total_az)
        for body in bodies:
            ax, ay, az = Acceleration[body]
            # Update Velocity
            body.vx += ax * timestep  
            body.vy += ay * timestep  
            body.vz += az * timestep
            # Update Position
            body.px += body.vx * timestep
            body.py += body.vy * timestep
            body.pz += body.vz * timestep
            #center = [body.px, body.py, body.pz]
            #radius = [body.re]
            #u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
            #x = radius*np.cos(u)*np.sin(v)
            #y = radius*np.sin(u)*np.sin(v)
            #z = radius*np.cos(v)
            #aaa.plot_surface(x-center[0], y-center[1], z-center[2], color=body.color, label=body.name)
            aaa.plot3D(body.px, body.py, body.pz, marker='.',linestyle='-', color=body.color, label=body.name)
            handles, labels = aaa.get_legend_handles_labels()  
            lgd = dict(zip(labels, handles))
            aaa.legend(lgd.values(), lgd.keys())
            plt.pause(0.001)
            fig.canvas.draw()

            











# Defining Objects
# Please note that planets constants may be incorrect or incomplete and initial positions and velocities aren't exact values.
def main():
    sun = Body()
    sun.name = 'Sun'
    sun.mass = 1.98892 * 10 ** 30
    sun.re = 696340 * 1000
    sun.color='yellow'

    mercury = Body()
    mercury.name = 'Mercury'
    mercury.mass = 0.33011 * 10 ** 24
    mercury.re = 2439.7 * 1000
    mercury.j2 = 0.00006
    mercury.py = 0.387 * AU
    mercury.vx = -47.36 * 1000
    mercury.color='black'

    venus = Body()
    venus.name = 'Venus'
    venus.mass = 4.8685 * 10 ** 24
    venus.re = 6051.8 * 1000
    venus.j2 = 0.000027
    venus.px = 0.723 * AU
    venus.vy = 35.02 * 1000
    venus.color='green'

    earth = Body()
    earth.name = 'Earth'
    earth.mass = 5.9742 * 10 ** 24
    earth.j2 = 1.0826269e-03
    earth.j3 = -2.5323000e-06
    earth.j4 = -1.6204000e-06
    earth.j5 = -0.15 * 10 ** 6
    earth.j6 = 0.57 * 10 ** 6
    earth.re = 6378136.3
    earth.px = -1 * AU
    earth.vy = -29.783 * 1000
    earth.color='blue'


    mars = Body()
    mars.name = 'Mars'
    mars.mass = 0.64171 * 10 ** 24
    mars.re = 3389.5 * 1000
    mars.j2 = 0.001964
    mars.j3 = 0 * 0.000036
    mars.py = -1.524 * AU
    mars.vx = 24.07 * 1000
    mars.color='red'


    jupiter = Body()
    jupiter.name = 'Jupiter'
    jupiter.mass = 1898.19 * 10 ** 24
    jupiter.re = 69911 * 1000
    jupiter.j2 = 0.01475
    jupiter.j3 = 0
    jupiter.j4 = -0.00058
    jupiter.j5 = 0
    jupiter.j6 = 3.4 * 10 ** -5
    jupiter.py = 5.204 * AU
    jupiter.vx = -13.06 * 1000
    jupiter.color='orange'


    saturn = Body()
    saturn.name = 'Saturn'
    saturn.mass = 568.34 * 10 ** 24
    saturn.re = 58232 * 1000
    saturn.j2 = 0.01645
    saturn.j3 = 0
    saturn.j4 = -0.000871
    saturn.j5 = 0
    saturn.j6 = 7 * 10 ** -5
    saturn.px = 9.582 * AU
    saturn.vy = 9.68 * 1000
    saturn.color='violet'


    uranus = Body()
    uranus.name = 'Uranus'
    uranus.mass = 86.813 * 10 ** 24
    uranus.re = 25362 * 1000
    uranus.j2 = 0.012
    uranus.px = -19.201 * AU
    uranus.vy = -6.8 * AU
    uranus.color='black'


    neptune = Body()
    neptune.name = 'Neptune'
    neptune.mass = 102.413 * 10 ** 24
    neptune.re = 24622 * 1000
    neptune.j2 = 0.004
    neptune.py = -30.047 * AU
    neptune.vx = 5.43 * 1000
    neptune.color='blue'


    loop([sun, mercury, venus, earth, mars])#, jupiter, saturn, uranus, neptune]) # Here we can choose which objects we want to simulate for example here we choose Sun, Mercury, Venus, Earth and Mars 



if __name__ == '__main__':
    main()


