import os, sys, subprocess
import optparse
from itertools import cycle
from lxml import etree

data_dir = os.path.join(os.getcwd(), 'data')
# import python modules from the $SUMO_HOME/tools directory

sumo_root = '/usr/share/sumo'
try:
    sumo_home = os.path.join(sumo_root, 'tools')
    sys.path.append(sumo_home)  # tutorial in docs
    print(sys.path)
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

# the port used for communicating with sumo instance
PORT = 8873


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


def gen_network(netname, xnumber, ynumber, xlength, ylength, nettype='grid', tlstype='actuated'):
    """
    Generate network using netgenerate tool. Currently support grid net only.
    """
    net_dir = os.path.join(data_dir, netname)
    if not os.path.isdir(net_dir):
        os.makedirs(net_dir)
    netgenerate = checkBinary('netgenerate')
    netfile = os.path.join(net_dir, netname + '.net.xml')
    netgenProcessor = subprocess.Popen([netgenerate, '--%s' % nettype,
                                        '--grid.x-number', xnumber, '--grid.y-number', ynumber,
                                        '--grid.x-length', xlength, '--grid.y-length', ylength,
                                        '--tls.guess', 'true', '--tls.default-type', tlstype,
                                        '--default.lanenumber', '2',
                                        '--%s.attach-length' % nettype, xlength,
                                        '-o', netfile], stdout=sys.stdout, stderr=sys.stderr)
    return netfile


def gen_randomtrips(netname, edgeprob, endtime='3600', seed='42',
                    trip_attrib="departLane=\"best\" departSpeed=\"max\" departPos=\"random\""):
    net_dir = os.path.join(data_dir, netname)
    netfile = os.path.join(net_dir, netname + '.net.xml')
    tripfile = os.path.join(net_dir, netname + '.trip.xml')
    roufile = os.path.join(net_dir, netname + '.rou.xml')
    rantrip_generator = os.path.join(sumo_home, 'randomTrips.py')
    gentripProcessor = subprocess.Popen([rantrip_generator, '-n', netfile,
                                         '-e', endtime, '-s', seed,
                                         '--fringe-factor', edgeprob, '--trip-attributes', trip_attrib,
                                         '-o', tripfile, '-r', roufile], stdout=sys.stdout, stderr=sys.stderr)
    return roufile


def gen_detectors(netname):
    net_dir = os.path.join(data_dir, netname)
    netfile = os.path.join(net_dir, netname + '.net.xml')
    e1file = os.path.join(net_dir, netname + '_e1.add.xml')
    e2file = os.path.join(net_dir, netname + '_e2.add.xml')
    e3file = os.path.join(net_dir, netname + '_e3.add.xml')
    e1generator = os.path.join(sumo_home, 'output', 'generateTLSE1Detectors.py')
    e2generator = os.path.join(sumo_home, 'output', 'generateTLSE2Detectors.py')
    e3generator = os.path.join(sumo_home, 'output', 'generateTLSE3Detectors.py')
    e_files = [e1file, e2file, e3file]
    e_generators = [e1generator, e2generator, e3generator]
    paras = zip(e_generators, cycle(['-n']), cycle([netfile]), cycle(['-o']), e_files)
    # detectors = list()
    for p in paras:
        p = list(p)
        d = subprocess.Popen(p, stdout=sys.stdout, stderr=sys.stderr)
        # detectors.append(d)
    # detectors = map(subprocess.Popen, paras)
    return e_files


def gen_sumocfg(netname):
    sumocfg_file = os.path.join(data_dir, netname, netname+'.sumo.cfg')
    conf_root = etree.Element("configuration", nsmap={'xsi': "http://www.w3.org/2001/XMLSchema-instance"})
    # conf_root.set('xsi:noNamespaceSchemaLocation', "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
    # Set Input file
    conf_input = etree.SubElement(conf_root, 'input')
    netfile = netname + '.net.xml'
    roufile = netname + '.rou.xml'
    detectors = [netname + '_e%d.add.xml' % (i + 1) for i in range(3)]
    input_netfile = etree.SubElement(conf_input, 'net-file')
    input_netfile.set('value', netfile)
    input_roufile = etree.SubElement(conf_input, 'route-files')
    input_roufile.set('value', roufile)
    input_addfile = etree.SubElement(conf_input, 'additional-files')
    input_addfile.set('value', " ".join(detectors))
    # Set Output file
    output_dir = os.path.join(data_dir, netname, 'output')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    conf_output = etree.SubElement(conf_root, 'output')
    netstatfile = 'output/'+netname+'_netstate.sumo.tr'
    tripinfo = 'output/'+netname+'_tripinfo.xml'
    vehroute = 'output/'+netname+'_vehroutes.xml'
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
    conf_tree.write(sumocfg_file, pretty_print=True, xml_declaration=True, encoding='utf-8')







if __name__ == '__main__':
    options = get_options()
    # Raise SUMO
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # Generate a 3x3 intersections network
    netname = '3x3'
    netfile = gen_network(netname, '3', '3', '400', '400')
    roufile = gen_randomtrips(netname, '10')
    # print(roufile)
    detectors = gen_detectors(netname)
    # print(detectors)
    gen_sumocfg(netname)
