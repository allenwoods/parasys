import optparse
import os
import sys
import subprocess

# SET SUMO HOME
os.environ.setdefault('SUMO_HOME', '/usr/share/sumo')

from src.env import SUMOENV
from src.mission import TrafficSim

data_dir = os.path.join(os.getcwd(), 'data')


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


if __name__ == '__main__':
    options = get_options()
    # Generate a 3x3 intersections network
    env_1x1_static = SUMOENV(data_dir, '1x1_static', 1, 1, tlstype='static')
    env_1x1_actuated = SUMOENV(data_dir, '1x1_actuated', 1, 1, tlstype='actuated')
    env_3x3_static = SUMOENV(data_dir, '3x3_static', 3, 3, tlstype='static')
    env_3x3_actuated = SUMOENV(data_dir, '3x3_actuated', 3, 3, tlstype='actuated')
    envs = [env_1x1_static, env_1x1_actuated, env_3x3_static, env_3x3_actuated]
    for e in envs:
        if not e.iscreated():
            e.create()
    # the port used for communicating with sumo instance
    PORT = 8873

    # Raise SUMO
    sumoProcess = env_3x3_static.run(PORT)
    get_tls_status = TrafficSim(PORT)
    get_tls_status.run()
    sumoProcess.wait()
