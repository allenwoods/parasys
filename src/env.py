import os
import subprocess
import sys
from itertools import cycle

from lxml import etree

sumo_root = '/usr/share/sumo'

try:
    sumo_home = os.path.join(sumo_root, 'tools')
    sys.path.append(sumo_home)  # tutorial in docs
    print(sys.path)
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")


class SUMOENV:
    def __init__(self, data_dir, netname, xnumber, ynumber,
                 xlength=400, ylength=400, nettype='grid', tlstype='actuated', rouprob=10, steps=3600):
        """
        Initialize SUMO environment.
        :param data_dir: where XML saved
        :param netname: name of network
        :param xnumber: nodes' number on x axis
        :param ynumber: nodes' number on y axis
        :param xlength: length of each edge on x axis
        :param ylength: length of each edge on y axis
        :param nettype: type of network, sumo support 'grid', 'spider', 'random'.
        :param tlstype: style of traffic light(static, actuated)
        :param rouprob: probability to generate a route begin/end at edge without precursor/successor
        """
        if xnumber < 2 or ynumber < 2:
            raise ValueError("The number of nodes must be at least 2 in both directions.")
        self.netname = netname
        self.data_dir = data_dir
        self.net_dir = os.path.join(self.data_dir, self.netname)
        if not os.path.isdir(self.net_dir):
            os.makedirs(self.net_dir)
        self.output_dir = os.path.join(self.net_dir, 'output')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        self.xnumber = str(xnumber)
        self.ynumber = str(ynumber)
        self.xlength = str(xlength)
        self.ylength = str(ylength)
        self.rouprob = rouprob
        self.tlstype = tlstype
        self.nettype = nettype
        self.steps = str(steps)
        self.netfile = os.path.join(self.net_dir, self.netname + '.net.xml')
        self.tripfile = os.path.join(self.net_dir, self.netname + '.trip.xml')
        self.roufile = os.path.join(self.net_dir, self.netname + '.rou.xml')
        self.detectors = [os.path.join(self.net_dir, self.netname + '_e%d.add.xml' % (i + 1))
                          for i in range(3)]
        self.sumocfg = os.path.join(self.data_dir, self.netname, self.netname + '.sumo.cfg')
        self.sumocfg_nodet = os.path.join(self.data_dir, self.netname, self.netname + '_nodet.sumo.cfg')

    def create(self):
        """
        Create the XML needed for simulation
        :return:
        """
        self.gen_network(self.xnumber, self.ynumber, self.xlength, self.ylength,
                         nettype=self.nettype, tlstype=self.tlstype)  # Set length to 400
        self.gen_randomtrips(self.rouprob, endtime=self.steps)  # Set edge prop to 10
        self.gen_detectors()
        self.sumocfg = self.gen_sumocfg()
        self.sumocfg_nodet = self.gen_sumocfg(withdetector=False)

    def iscreated(self):
        """
        Check if the XMLs have been created
        :return:
        """
        return os.path.isfile(self.sumocfg)

    def run(self, port, gui=False, withdet=True):
        """
        Raise SUMO and run the simulation.
        :param port: remote-port
        :param gui: with GUI
        :param withdet: with detector(e1,e2,e3)
        :return:
        """
        if gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')
        if withdet:
            sumocfg = self.sumocfg
        else:
            sumocfg = self.sumocfg_nodet

        sumo_env = os.environ.copy()
        sumo_env['SUMO_HOME'] = sumo_root
        # set_sumo_home = 'env SUMO_HOME=%s' % sumo_home
        sumoProcess = subprocess.Popen([sumoBinary, '-c', sumocfg, '--remote-port', str(port)],
                                       env=sumo_env, stdout=sys.stdout, stderr=sys.stderr)
        # sumoProcess.wait()
        return sumoProcess

    def gen_network(self, xnumber, ynumber, xlength, ylength,
                    nettype='grid', tlstype='actuated'):
        """
        Generate network model
        :param xnumber: nodes on x axis
        :param ynumber: nodes on y axis
        :param xlength: length of each edge on x axis
        :param ylength: length of each edge on y axis
        :param nettype: type of network, sumo support 'grid', 'spider', 'random'.
        :param tlstype:
        :return:
        """
        netgenerate = checkBinary('netgenerate')
        netgenProcessor = subprocess.Popen([netgenerate, '--%s' % nettype,
                                            '--grid.x-number', xnumber, '--grid.y-number', ynumber,
                                            '--grid.x-length', xlength, '--grid.y-length', ylength,
                                            '--tls.guess', 'true', '--tls.default-type', tlstype,
                                            '--default.lanenumber', '2',
                                            '--%s.attach-length' % nettype, xlength,
                                            '-o', self.netfile], stdout=sys.stdout, stderr=sys.stderr)

    def gen_randomtrips(self, rouprob, endtime, seed='42',
                        trip_attrib="departLane=\"best\" departSpeed=\"max\" departPos=\"random\""):
        """
        Generate random trip on given network
        Warning: Sometimes routefile might not be created
        :param rouprob: probability to generate a route begin/end at edge without precursor/successor
        :param endtime: end time of the simulation
        :param seed: random seed
        :param trip_attrib: additional attribute of random trips
        :return:
        """
        rantrip_generator = os.path.join(sumo_home, 'randomTrips.py')
        gentripProcessor = subprocess.Popen([rantrip_generator, '-n', self.netfile,
                                             '-e', endtime, '-s', seed,
                                             '--fringe-factor', rouprob, '--trip-attributes', trip_attrib,
                                             '-o', self.tripfile, '-r', self.roufile], stdout=sys.stdout,
                                            stderr=sys.stderr)

    def gen_detectors(self):
        """
        Generate Detector on the given network
        :return:
        """
        e1generator = os.path.join(sumo_home, 'output', 'generateTLSE1Detectors.py')
        e2generator = os.path.join(sumo_home, 'output', 'generateTLSE2Detectors.py')
        e3generator = os.path.join(sumo_home, 'output', 'generateTLSE3Detectors.py')
        det_outputs = ['output/' + 'e%d_output.xml' % (i + 1) for i in range(3)]
        e_generators = [e1generator, e2generator, e3generator]
        paras = zip(e_generators, cycle(['-n']), cycle([self.netfile]),
                    cycle(['-o']), self.detectors, cycle(['-r']), det_outputs)
        # detectors = list()
        for p in paras:
            p = list(p)
            d = subprocess.Popen(p, stdout=sys.stdout, stderr=sys.stderr)
            # detectors.append(d)
            # detectors = map(subprocess.Popen, paras)

    def gen_sumocfg(self, withdetector=True):
        """
        Generate sumo.cfg file for sumo to load
        :param withdetector: controll whether the detector should be add in the file or not
        :return:
        """
        conf_root = etree.Element("configuration", nsmap={'xsi': "http://www.w3.org/2001/XMLSchema-instance"})
        # conf_root.set('xsi:noNamespaceSchemaLocation', "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
        # Set Input file
        conf_input = etree.SubElement(conf_root, 'input')
        netfile = self.netname + '.net.xml'  # Use relative address
        roufile = self.netname + '.rou.xml'  # makes configuaration portable
        detectors = [self.netname + '_e%d.add.xml' % (i + 1) for i in range(3)]
        input_netfile = etree.SubElement(conf_input, 'net-file')
        input_netfile.set('value', netfile)
        input_roufile = etree.SubElement(conf_input, 'route-files')
        input_roufile.set('value', roufile)
        if withdetector:
            input_addfile = etree.SubElement(conf_input, 'additional-files')
            input_addfile.set('value', " ".join(detectors))
        # Set Output file
        conf_output = etree.SubElement(conf_root, 'output')
        netstatfile = 'output/' + self.netname + '_netstate.sumo.tr'
        tripinfo = 'output/' + self.netname + '_tripinfo.xml'
        vehroute = 'output/' + self.netname + '_vehroutes.xml'
        output_nets = etree.SubElement(conf_output, 'netstate-dump')
        output_nets.set('value', netstatfile)
        output_tripinfo = etree.SubElement(conf_output, 'tripinfo-output')
        output_tripinfo.set('value', tripinfo)
        output_vehroute = etree.SubElement(conf_output, 'vehroute-output')
        output_vehroute.set('value', vehroute)
        # Set Time
        conf_time = etree.SubElement(conf_root, 'time')
        time_begin = etree.SubElement(conf_time, 'begin')
        time_begin.set('value', '0')
        time_end = etree.SubElement(conf_time, 'end')
        time_end.set('value', '3600')
        time_roustep = etree.SubElement(conf_time, 'route-steps')
        time_roustep.set('value', '-1')
        # Set Time to teleport
        conf_teleport = etree.SubElement(conf_root, 'time-to-teleport')
        conf_teleport.set('value', '-1')

        # Write to sumo.cfg
        conf_tree = etree.ElementTree(conf_root)
        if withdetector:
            sumocfg_file = self.sumocfg
        else:
            sumocfg_file = self.sumocfg_nodet
        conf_tree.write(sumocfg_file, pretty_print=True, xml_declaration=True, encoding='utf-8')
        return sumocfg_file
