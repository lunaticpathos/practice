import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

g0 = 9.8
Re = 6371000

boosterMass = 56200
shroudMass = 230
rvMass = 1280
dia = 2.1
launchAngle = 89

stgmass = [boosterMass*.75, boosterMass*.20, boosterMass*.05]
fuelmass = [stgmass[0]*.92, stgmass[1]*.91, stgmass[2]*.87]
Is = [276, 296, 296]
T = [1610000, 456000, 177000]
burnTime = [65, 65, 40]
ballC = 120000
length = 1.5
noseR = 0.04
baseR = 0.26
halfAngle = 8.5*np.pi/180
noseH = (baseR - noseR)/np.tan(halfAngle)
rvArea = np.pi*(noseR**2) + np.pi*(noseR+baseR)*np.sqrt((noseH**2)+(baseR-noseR)**2)

def poweredFlight(t, y, thrust, area, specImpulse, dragRegime):
    [velocity, flightPathAngle, rangeAngle, height, mass] = y
    density = getAirDensity(height)
    if dragRegime == 1:
        Cd = getMissileDragCoeff(velocity, height)
    if dragRegime == 2:
        Cd = getRVDragCoeff(mass)
    gravityAccel = getGravityAccel(height)
    dVelocity = thrust/mass - Cd*density*(velocity**2)*area/(2*mass) - gravityAccel*np.sin(flightPathAngle)
    dRangeAngle = velocity*np.cos(flightPathAngle)/(Re + height)
    dFlightPathAngle = dRangeAngle - gravityAccel*np.cos(flightPathAngle)/velocity
    dHeight = velocity*np.sin(flightPathAngle)
    dMass = -thrust/(g0*specImpulse)
    return [dVelocity, dFlightPathAngle, dRangeAngle, dHeight, dMass]

def getAirDensity(height):
    if height < 11000:
        density = 1.2985 - (1.2985-0.3639)*height/11000
    elif height < 20000:
        density = 0.3639 - (0.3639-0.0880)*(height-11000)/9000
    elif height < 32000:
        density = 0.0880 - (0.0880-0.0105)*(height-20000)/12000
    elif height < 47000:
        density = 0.0105 - (0.0105-0.0020)*(height-32000)/15000
    elif height < 51000:
        density = 0.0020
    else:
        density = 0
    return density

def getMissileDragCoeff(velocity, height):
    if height < 11000:
        temp = 292 - 6.5*height/1000
    elif height < 20000:
        temp = 216.5
    elif height < 32000:
        temp = 216.5 + (height-20000)/1000
    elif height < 47000:
        temp = 228.5 + 2.8*(height-32000)/1000
    elif height < 51000:
        temp = 270.5
    elif height < 71000:
        temp = 270.5 - 2.8*(height-51000)/1000
    elif height < 84852:
        temp = 214.5 - 2*(height-71000)/1000
    else:
        temp = 273 - 86.28
    c = np.sqrt(1.398*8.3145*temp/28.9645)
    M = velocity/c
    if M < 0.7:
        Cd = 0.15
    elif M < 1.5:
        Cd = 0.15 + (0.425-0.15)*(M-0.7)/0.8
    elif M < 2:
        Cd = 0.425 - (0.425-0.25)*(M-1.5)/0.5
    elif M < 4.5:
        Cd = 0.25 - (0.25-0.15)*(M-2)/2.5
    else:
        Cd = 0.15
    return Cd

def getRVDragCoeff(mass):
    return mass*g0/(ballC*rvArea)

def getGravityAccel(height):
    return g0*(Re**2)/((height + Re)**2)

y0 = [1, launchAngle*np.pi/180, 0, 0, boosterMass + shroudMass + rvMass]
t0 = 0

r = ode(poweredFlight).set_integrator('zvode', method='bdf')
r.set_initial_value(y0, t0).set_f_params(T[0], np.pi*((dia/2)**2), Is[0], 1)
t1 = burnTime[0]
dt = 0.01

time = []
v = []
gamma = []
psi = []
psirad = []
h = []
m = []

while r.successful() and r.t < t1 and r.y[3] > -100:
    time.append(r.t)
    r.integrate(r.t+dt)
    v.append(r.y[0])
    gamma.append(r.y[1]*180/np.pi)
    psi.append(r.y[2]*180/np.pi)
    psirad.append(r.y[2])
    h.append(r.y[3])
    m.append(r.y[4])

y0 = r.y
y0[4] = boosterMass + shroudMass + rvMass - stgmass[0]
t0 = r.t

r = ode(poweredFlight).set_integrator('zvode', method='bdf')
r.set_initial_value(y0, t0).set_f_params(T[1], np.pi*((dia/2)**2), Is[1], 1)

t1 = burnTime[0] + burnTime[1]

while r.successful() and r.t < t1 and r.y[3] > 0:
    time.append(r.t)
    r.integrate(r.t+dt)
    v.append(r.y[0])
    gamma.append(r.y[1]*180/np.pi)
    psi.append(r.y[2]*180/np.pi)
    psirad.append(r.y[2])
    h.append(r.y[3])
    m.append(r.y[4])

y0 = r.y
y0[4] = boosterMass + shroudMass + rvMass - stgmass[0] - stgmass[1]
t0 = r.t

r = ode(poweredFlight).set_integrator('zvode', method='bdf')
r.set_initial_value(y0, t0).set_f_params(T[2], np.pi*((dia/2)**2), Is[2], 1)

t1 = burnTime[0] + burnTime[1] + burnTime[2]

while r.successful() and r.t < t1 and r.y[3] > 0:
    time.append(r.t)
    r.integrate(r.t+dt)
    v.append(r.y[0])
    gamma.append(r.y[1]*180/np.pi)
    psi.append(r.y[2]*180/np.pi)
    psirad.append(r.y[2])
    h.append(r.y[3])
    m.append(r.y[4])

y0 = r.y
y0[4] = boosterMass + shroudMass + rvMass - stgmass[0] - stgmass[1] - stgmass[2] - shroudMass
t0 = r.t

r = ode(poweredFlight).set_integrator('zvode', method='bdf')
r.set_initial_value(y0, t0).set_f_params(0, np.pi*((dia/2)**2), 1, 2)

while r.successful() and r.y[3] > 0 and r.t < 5400:
    time.append(r.t)
    r.integrate(r.t+dt)
    v.append(r.y[0])
    gamma.append(r.y[1]*180/np.pi)
    psi.append(r.y[2]*180/np.pi)
    psirad.append(r.y[2])
    h.append(r.y[3])
    m.append(r.y[4])

radius = []

for height in h:
    radius.append(Re + height)

radius = np.array(radius)

plt.plot(time, v)
plt.figure()
plt.plot(time, gamma)
plt.figure()
plt.plot(time, psi)
plt.figure()
plt.plot(time, h)
plt.figure()
plt.plot(time, m)
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(psirad, radius)
ax.plot(np.linspace(0,np.pi*2,256), np.linspace(Re,Re,256))
ax.grid(True)
plt.show()