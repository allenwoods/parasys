import os
import sys
import time
from src.log import timeit

sumo_root = os.environ.get('SUMO_HOME')

try:
    sumo_home = os.path.join(sumo_root, 'tools')
    sys.path.append(sumo_home)  # tutorial in docs
    from sumolib import checkBinary
    import traci
    from traci import simulationStep
    from traci import simulation
    from traci import trafficlights
    from traci import lane
except ImportError:
    sys.exit(
        'Please declare environment variable \'SUMO_HOME\' as the root directory '
        'of your sumo installation (it should contain folders \'bin\', \'tools\' and \'docs\')')


def run(port):
    """
    Initiate the TraCI Server
    :param port:
    :return:
    """
    traci.init(port)
    step = 0
    start = time.process_time()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        print(step)
        step += 1
    end = time.process_time()
    print("Total Running time: %f" % (end - start))
    traci.close()
    sys.stdout.flush()


class TrafficSim:
    def __init__(self, port, host='localhost', label='default'):
        self.port = port
        self.host = host
        self.label = label

    def run(self):
        traci.init(self.port, host=self.host)
        self.__get_model()
        while simulation.getMinExpectedNumber() > 0:
            self.step()

    @staticmethod
    def close():
        traci.close()
        sys.stdout.flush()

    def __get_model(self):
        self.tls = trafficlights.getIDList()
        self.tls_lane_pair = dict(zip(self.tls,
                                      [trafficlights.getControlledLanes(t) for t in self.tls]))

    @timeit
    def step(self):
        traci.simulationStep()
        tls_status = dict(zip(self.tls, map(trafficlights.getRedYellowGreenState, self.tls)))
        for t in self.tls:
            print(t)
            t_status = tls_status[t]
            t_status_list = [t_status[i:i + 5] for i in range(0, len(t_status), 5)]
            controlled_lanes = self.tls_lane_pair[t]
            N_lane = [controlled_lanes[1], controlled_lanes[3]]
            E_lane = [controlled_lanes[6], controlled_lanes[8]]
            S_lane = [controlled_lanes[11], controlled_lanes[13]]
            W_lane = [controlled_lanes[16], controlled_lanes[18]]
            t_lanes = N_lane + E_lane + S_lane + W_lane
            # Warning: get lanes status takes lots of time
            lanes_status = dict(zip(t_lanes, map(self.__get_lane_status, t_lanes)))
            t_l_staus = zip(t_status_list, [N_lane, E_lane, S_lane, W_lane])
            for i in t_l_staus:
                print(i)

    @staticmethod
    def __get_lane_status(laneid):
        halt_number = lane.getLastStepHaltingNumber(laneid)
        waiting_time = lane.getWaitingTime(laneid)
        return halt_number, waiting_time


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
