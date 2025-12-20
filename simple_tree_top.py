from mininet.net import Mininet
from mininet.topolib import TreeTopo
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info

def simple_test():
    """Create a simple tree topology for connectivity testing."""

    # 1. Create topology
    tree_topo = TreeTopo(depth=1, fanout=2)

    # 2. Create network and specify remote controller
    net = Mininet(topo=tree_topo, 
                  controller=RemoteController('c0', ip='127.0.0.1', port=6633), 
                  switch=OVSSwitch, 
                  autoSetMacs=True)

    info('*** Starting network\n')
    net.start()

    # 3. Wait for network and controller connection to stabilize
    info('*** Waiting for controller to connect...\n')
    net.waitConnected()

    # 4. Perform comprehensive connectivity test
    info('*** Testing network connectivity\n')
    net.pingAll()

    # 5. Enter CLI for manual testing
    info('*** Running CLI\n')
    CLI(net)

    # 6. Stop network after exiting
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    simple_test()