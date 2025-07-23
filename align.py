#Script to align axis of a 3D object to the world axes
from lsm6ds3 import LSM6DS3
import time

def read_acc(sent=0.000061):
  ax, ay, az, gx, gy, gz = lsm.get_readings()
  ax = ax * sent
  ay = ay * sent
  az = az * sent

  return ax, ay, az

def align(sent=0.1):
  global lsm
  lsm = LSM6DS3()

  x_avg = 0.180
  y_avg = -0.0214
  z_avg = 0.980

  x_align = None
  y_align = None
  z_align = None

  while x_align != True and y_align != True and z_align != True:
    ax, ay, az = read_acc()
    time.sleep(0.2)

    if x_avg - sent < ax < x_avg + sent:
      x_align = True
      print("X axis aligned")
    else:
      print("X axis not aligned, current value:", ax)
    if y_avg - sent < ay < y_avg + sent:
      y_align = True
      print("Y axis aligned")
    else:
      print("Y axis not aligned, current value:", ay)
    if z_avg - sent < az < z_avg + sent:
      z_align = True
      print("Z axis aligned")
    else:
      print("Z axis not aligned, current value:", az)
  
  print("Alignment complete")

