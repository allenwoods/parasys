from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from sim.traffic import enviroment   # Change name here to your module!
sys.path.pop(0)

def car(env):
     while True:
         print('Start parking at %d' % env.now)
         parking_duration = 5
         yield env.timeout(parking_duration)

         print('Start driving at %d' % env.now)
         trip_duration = 2
         yield env.timeout(trip_duration)

if __name__ == '__main__':
    env = enviroment.traffic_env()
    env.process(car(env))
    env.run(until=15)