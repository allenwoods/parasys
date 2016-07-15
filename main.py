import optparse
import os
import sys
import subprocess

from src.env import SUMOENV
from src.mission import TrafficSim

data_dir = os.path.join(os.getcwd(), 'data')
# import python modules from the $SUMO_HOME/tools directory
# sumo_root = '/usr/share/sumo'
# try:
#     sumo_home = os.path.join(sumo_root, 'tools')
#     sys.path.append(sumo_home)  # tutorial in docs
#     print(sys.path)
#     from sumolib import checkBinary
# except ImportError:
#     sys.exit(
#         "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


# def run(port):
#     """
#     Initiate the TraCI Server
#     :param port:
#     :return:
#     """
#     import traci
#     traci.init(port)
#     step = 0
#     while traci.simulation.getMinExpectedNumber() > 0:
#         traci.simulationStep()
#         print(step)
#         step += 1
#     traci.close()
#     sys.stdout.flush()

if __name__ == '__main__':
    options = get_options()
    # Generate a 3x3 intersections network
    sumo_env_static = SUMOENV(data_dir, '3x3_static', 3, 3, tlstype='static')
    sumo_env_actuated = SUMOENV(data_dir, '3x3_actuated', 3, 3, tlstype='actuated')
    # the port used for communicating with sumo instance
    PORT = 8873

    # Raise SUMO
    if not sumo_env_static.iscreated():
        sumo_env_static.create()
        sumo_env_actuated.create()
    sumoProcess = sumo_env_static.run(PORT)
    get_tls_status = TrafficSim(PORT)
    get_tls_status.run()
    sumoProcess.wait()
