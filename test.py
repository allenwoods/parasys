import optparse
import os
from time import strftime as current_time
from SumoEnv.environ import TrafficSim
from SumoEnv.simulation import SumoEnv


# SET SUMO HOME
os.environ.setdefault('SUMO_HOME', '/usr/share/sumo')

DATA_DIR = os.path.join(os.getcwd(), 'data')
CFG_DIR = os.path.join(DATA_DIR, 'cfg')
NET_DIR = os.path.join(DATA_DIR, 'network')
SUMMARY_DIR = os.path.join(DATA_DIR, 'summary')


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


if __name__ == '__main__':
    mission_start_time = current_time('%Y%m%d%H%M%S')
    task_cfg_dir = os.path.join(CFG_DIR, 'a3c')
    env = SumoEnv(task_cfg_dir, mission_start_time,
                  xnumber=3, ynumber=3, gui=False)
    task = TrafficSim(env)
    s_t = task.get_initial_state()
    s_t1, r_t, terminated, info = task.step(1)
    print(s_t1)
    print(s_t1.shape)
    # print("S_t1%s" % str(s_t1))
    # print("R_t1%s" % str(r_t))
    # print("Terminated:%s" % str(terminated))

    # sumo, x_0 = env.reset()
    # print(x_0)
    # options = get_options()
    # # Generate a 3x3 intersections network
    # env_1x1_static = SumoCfg(data_dir, '1x1_static', 1, 1, tls_type='static')
    # env_1x1_actuated = SumoCfg(data_dir, '1x1_actuated', 1, 1, tls_type='actuated')
    # env_3x3_static = SumoCfg(data_dir, '3x3_static', 3, 3, tls_type='static')
    # env_3x3_actuated = SumoCfg(data_dir, '3x3_actuated', 3, 3, tls_type='actuated')
    # envs = [env_1x1_static, env_1x1_actuated, env_3x3_static, env_3x3_actuated]
    # for e in envs:
    #     if not e.iscreated():
    #         e.make()
    # # the port used for communicating with sumo instance
    # PORT = 8875
    #
    # # Raise SUMO
    # mission_start_time = current_time('%Y%m%d%H%M%S')
    # for e in envs:
    #     sumoProcess = e.start(PORT, mission_start_time, gui=True)
    #     get_tls_status = TrafficSim(PORT)
    #     logger = get_tls_status.run(update_steps=15)
    #     get_tls_status.close()
    #     # sumoProcess.wait()
    #     logger.to_csv(os.path.join(e.task_record_dir, e.netname + '_log.csv'), index=False)

    # env_1x1_halts = SumoEnv(data_dir, '1x1_haltest', 1, 1, tls_type='static')
    # env_3x3_halts = SumoEnv(data_dir, '3x3_haltest', 1, 1, tls_type='static')
    # halt_envs = [env_1x1_halts, env_3x3_halts]
    # for e in halt_envs:
    #     if not e.iscreated():
    #         e.create()
    # # the port used for communicating with sumo instance
    # PORT = 8875

    # Raise SUMO
    # sumoProcess = env_1x1_static.init(PORT, 'haltest', gui=True)
    # for e in halt_envs:
    #     for stepsize in range(15, 31):
    #         mission_start_time = current_time('%Y%m%d%H%M%S')
    #         sumoProcess = e.init(PORT, mission_start_time)
    #         get_tls_status = TrafficSim(PORT)
    #         logger = get_tls_status.run(update_steps=stepsize, strategy='lazy_police')
    #         get_tls_status.close()
    #         # sumoProcess.wait()
    #         logger.to_csv(os.path.join(e.task_record_dir, e.netname +
    #                                    '_step%s_log.csv' % str(stepsize)), index=False)
