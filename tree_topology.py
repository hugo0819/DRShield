from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info

from flask import Flask, request, jsonify
import threading
import requests
import time
import config

# --- Global Variables ---
net = None
EPISODE_LENGTH = config.EPISODE_LENGTH
VICTIM_IP = config.DEST_IP
SPOOFED_IP = config.SPOOFED_SRC_IP
CONTROLLER_IP = '127.0.0.1'
CONTROLLER_PORT = 8080 # Default port for Ryu Web API

# --- Flask API Application ---
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok"}, 200

@app.route('/start_episode', methods=['POST'])
def start_episode():
    data = request.get_json()
    attacker_id = data.get('attacker_id', 0)
    benign_id = data.get('benign_id', 1)

    if not net or attacker_id >= len(net.hosts) or benign_id >= len(net.hosts):
        return jsonify({'status': 'error', 'message': 'Invalid host ID or network not ready'}), 400

    # Get participants
    attacker_host = net.hosts[attacker_id]
    benign_host = net.hosts[benign_id]
    victim_host = net.get(config.DEST_NAME) # Get victim host by name h8

    if attacker_host == benign_host:
        benign_host = net.hosts[(benign_id + 1) % len(net.hosts)]

    info(f"--- Starting Episode ---\n")
    info(f"Attacker: {attacker_host.name} -> {victim_host.name} (UDP Flood)\n")
    info(f"Benign Traffic: {attacker_host.name} -> {benign_host.name} (Ping)\n")

    # --- Traffic Generation (Final Adjustment: Use iperf and ping) ---

    # 1. Start iperf server on victim
    #    -s: Server mode, -u: UDP, -P 0: Disable multithreading for simplicity
    victim_host.cmd('iperf -s -u -P 0 > /dev/null 2>&1 &')
    # Give server some time to start
    time.sleep(1)

    # 2. Attack traffic: Attacker acts as iperf client, launching UDP flood to victim
    #    -c <ip>: Client mode, connect to server
    #    -u: UDP mode
    #    -b 100M: Bandwidth set to 100Mbps (simulate flood)
    #    -t <secs>: Duration
    attack_cmd = (f'iperf -c {victim_host.IP()} -u -b 100M '
                  f'-t {EPISODE_LENGTH - 2} > /dev/null 2>&1 &')

    # 3. Normal traffic: Generate ping between attacker and another benign user
    benign_cmd = (f'ping -c 5 -i 1 {benign_host.IP()} > /dev/null 2>&1 &')

    # Execute commands
    attacker_host.cmd(attack_cmd)
    attacker_host.cmd(benign_cmd)

    return jsonify({'status': 'success'})

def run_api_server():
    """Run Flask server in a separate thread."""
    info('*** Starting traffic control API on http://127.0.0.1:5001\n')
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

def wait_for_controller_ready():
    """
    Phase 2: Wait for controller to confirm topology readiness.
    Poll the controller's /ready endpoint until it returns True.
    """
    ready_url = f"http://{CONTROLLER_IP}:{CONTROLLER_PORT}/ready"
    info('*** Waiting for controller to confirm topology discovery...\n')
    while True:
        try:
            response = requests.get(ready_url, timeout=2)
            if response.status_code == 200 and response.json().get('ready'):
                info('*** Controller has confirmed topology is ready.\n')
                return
        except requests.exceptions.RequestException:
            info('... controller not ready yet, retrying in 2s.\n')
        time.sleep(2)

def create_and_run_network():
    """Main function: Create network, start API, and enter CLI."""
    global net

    net = Mininet(controller=RemoteController, switch=OVSSwitch, autoSetMacs=True)
    net.addController('c0')

    info('*** Adding hosts and switches\n')
    # (Topology construction code remains unchanged)
    switches = {f's{i}': net.addSwitch(f's{i}') for i in range(1, 8)}
    hosts = {f'h{i}': net.addHost(f'h{i}', ip=f'10.0.0.{i}') for i in range(1, 9)}
    info('*** Creating links\n')
    net.addLink(switches['s1'], switches['s2']); net.addLink(switches['s1'], switches['s3'])
    net.addLink(switches['s2'], switches['s4']); net.addLink(switches['s2'], switches['s5'])
    net.addLink(switches['s3'], switches['s6']); net.addLink(switches['s3'], switches['s7'])
    net.addLink(switches['s4'], hosts['h1']); net.addLink(switches['s4'], hosts['h2'])
    net.addLink(switches['s5'], hosts['h3']); net.addLink(switches['s5'], hosts['h4'])
    net.addLink(switches['s6'], hosts['h5']); net.addLink(switches['s6'], hosts['h6'])
    net.addLink(switches['s7'], hosts['h7']); net.addLink(switches['s7'], hosts['h8'])

    info('*** Starting network\n')
    net.start()

    # Phase 1: Network readiness and controller warm-up
    info('*** Phase 1: Network is up. Warming up controller for discovery...\n')
    # Use more reliable pingAllFull with increased timeout to ensure sufficient discovery events
    # net.pingAllFull(timeout="2") # Error: pingAllFull does not accept timeout parameter
    info('*** Pinging all hosts to trigger discovery...\n')
    net.pingAll() # Correct: Use pingAll to trigger ARP and initial packets

    # Perform additional arping to ensure all hosts are seen as sources
    info('*** Forcing final ARP discovery...\n')
    for host in net.hosts:
        host.cmd('arping -c 1 10.0.0.254 > /dev/null 2>&1 &')
    time.sleep(3) # Wait for final arping commands to be sent

    # Phase 2: Wait for controller confirmation
    wait_for_controller_ready()

    # Phase 3: Start traffic API and enter CLI
    info('*** Phase 3: Starting traffic API and CLI.\n')
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()

    info('\n*** System is fully operational. Training can now begin.\n')
    CLI(net)

    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    create_and_run_network()