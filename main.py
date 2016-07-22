import optparse
import os
import sys
from time import strftime as current_time
import subprocess

# SET SUMO HOME
os.environ.setdefault('SUMO_HOME', '/usr/share/sumo')

from src.env import SumoEnv
from src.simulation import TrafficSim

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
    env_1x1_static = SumoEnv(data_dir, '1x1_static', 1, 1, tls_type='static')
    env_1x1_actuated = SumoEnv(data_dir, '1x1_actuated', 1, 1, tls_type='actuated')
    env_3x3_static = SumoEnv(data_dir, '3x3_static', 3, 3, tls_type='static')
    env_3x3_actuated = SumoEnv(data_dir, '3x3_actuated', 3, 3, tls_type='actuated')
    envs = [env_1x1_static, env_1x1_actuated, env_3x3_static, env_3x3_actuated]
    # envs = [env_1x1_static, env_3x3_static]
    for e in envs:
        if not e.iscreated():
            e.create()
    # the port used for communicating with sumo instance
    PORT = 8875

    # Raise SUMO
    # sumoProcess = env_1x1_static.init(PORT, 'haltest', gui=True)
    mission_start_time = current_time('%Y%m%d%H%M%S')
    for e in envs:
        sumoProcess = e.init(PORT, mission_start_time)
        get_tls_status = TrafficSim(PORT)
        logger = get_tls_status.run(update_steps=15)
        get_tls_status.close()
        # sumoProcess.wait()
        logger.to_csv(os.path.join(e.task_record_dir, e.netname + '_log.csv'), index=False)

    env_1x1_halts = SumoEnv(data_dir, '1x1_haltest', 1, 1, tls_type='static')
    env_3x3_halts = SumoEnv(data_dir, '3x3_haltest', 1, 1, tls_type='static')
    halt_envs = [env_1x1_halts, env_3x3_halts]
    for e in halt_envs:
        if not e.iscreated():
            e.create()
    # the port used for communicating with sumo instance
    PORT = 8875

    # Raise SUMO
    # sumoProcess = env_1x1_static.init(PORT, 'haltest', gui=True)
    for e in halt_envs:
        for stepsize in range(15, 31):
            mission_start_time = current_time('%Y%m%d%H%M%S')
            sumoProcess = e.init(PORT, mission_start_time)
            get_tls_status = TrafficSim(PORT)
            logger = get_tls_status.run(update_steps=stepsize, strategy='haltest')
            get_tls_status.close()
            # sumoProcess.wait()
            logger.to_csv(os.path.join(e.task_record_dir, e.netname +
                                       '_step%s_log.csv' % str(stepsize)), index=False)
