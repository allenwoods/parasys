import os
import sys
import time

sumo_root = '/usr/share/sumo'

try:
    sumo_home = os.path.join(sumo_root, 'tools')
    sys.path.append(sumo_home)  # tutorial in docs
    print(sys.path)
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

    def step(self):
        traci.simulationStep()
        tls_status = dict(zip(self.tls, map(trafficlights.getRedYellowGreenState, self.tls)))



    @staticmethod
    def __get_lane_status(laneid):
        halt_number = lane.getLastStepHaltingNumber(laneid)
        waiting_time = lane.getWaitingTime(laneid)
        return halt_number, waiting_time


