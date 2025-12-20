"""
SDN DDoS Defense Controller
Supports two reinforcement learning algorithms:
1. Traditional PPO (flat state vector)
2. GNN+PPO (graph-structured state)
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ipv4, ether_types
from ryu.topology import event as topo_event, api as topo_api
from ryu.ofproto import ofproto_v1_3

# Import Ryu's REST API related components
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

import numpy as np
import random
import requests
import config
from sdn_environment import SDNEnvironment

# Define a unique instance name for API registration
SDN_CONTROLLER_INSTANCE_NAME = 'sdn_controller_api'


class SDNController(app_manager.RyuApp):
    """
    SDN DDoS Defense Controller

    Features:
    - Network topology discovery and management
    - Flow table statistics collection
    - Reinforcement learning training (supports PPO and GNN+PPO)
    - DDoS attack detection and mitigation
    """
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(SDNController, self).__init__(*args, **kwargs)
        wsgi = kwargs['wsgi']
        wsgi.register(SdnRestApi, {SDN_CONTROLLER_INSTANCE_NAME: self})

        # --- Network State ---
        self.mac_to_port = {}
        self.datapaths = {}
        self.hosts = {}
        self.state = {}
        self.flow_stats = {}

        # --- Traffic Counters ---
        self.attack_count = 0
        self.benign_count = 0

        # --- Deployment and Detection ---
        self.deployed_ml_switches = set()
        self.detection_rate = 0.95

        # --- Training State ---
        self.topology_ready = False
        self.training_started = False
        self.training_initializing = False
        self.env = None
        self.agent = None

        # --- Statistics Synchronization ---
        self.stats_event = hub.Event()
        self.stats_replies_expected = 0
        
        # --- Algorithm Selection ---
        self.use_gnn_ppo = getattr(config, 'USE_GNN_PPO', False)
        self.logger.info(f"Using algorithm: {'GNN+PPO' if self.use_gnn_ppo else 'PPO'}")

    # ========== Expected Scale and Readiness Check ==========
    def _expected_switches(self) -> int:
        return getattr(config, "EXPECTED_SWITCHES", 7)

    def _expected_hosts(self) -> int:
        return getattr(config, "EXPECTED_HOSTS", 8)

    def _check_topology_ready(self):
        """Check if the topology is ready, and if so, start training initialization."""
        if self.topology_ready:
            return

        switches = topo_api.get_all_switch(self)
        hosts = topo_api.get_all_host(self)
        
        sw_ready = len(switches) >= self._expected_switches()
        host_ready = len(hosts) >= self._expected_hosts()

        if sw_ready and host_ready:
            self.topology_ready = True
            self.logger.info(f"*** Topology ready: {len(switches)} switches, {len(hosts)} hosts discovered. ***")
            self.logger.info("*** Starting training environment initialization... ***")
            hub.spawn(self._training_initializer)

    # ========== 1) Topology Event Handling ==========
    @set_ev_cls(topo_event.EventSwitchEnter)
    def _switch_enter_handler(self, ev):
        datapath = ev.switch.dp
        self.logger.info(f"Topology discovery: Switch {datapath.id} connected")
        self.datapaths[datapath.id] = datapath
        self.mac_to_port.setdefault(datapath.id, {})
        self._add_default_flow(datapath)
        self._check_topology_ready()

    @set_ev_cls(topo_event.EventHostAdd)
    def _host_add_handler(self, ev):
        host = ev.host
        self.logger.info(f"Topology discovery: Host {host.mac} (IP: {host.ipv4}) connected")
        self.hosts[host.mac] = host
        self._check_topology_ready()

    def _add_default_flow(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    # ========== 2) Switching Logic (Learning-based Forwarding)==========
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src

        # 1. Learn MAC-to-port mapping
        self.mac_to_port.setdefault(dpid, {})
        if src not in self.mac_to_port[dpid]:
            self.mac_to_port[dpid][src] = in_port

        # 2. Determine output port
        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        # 3. Install flow table for IP traffic
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            match = parser.OFPMatch(
                eth_type=ether_types.ETH_TYPE_IP,
                ipv4_src=ip_pkt.src,
                ipv4_dst=ip_pkt.dst,
                ip_proto=ip_pkt.proto
            )
            timeout = config.EPISODE_LENGTH * 2 if self.training_started else 0
            self.add_flow(datapath, 10, match, actions, idle_timeout=timeout)

        # 4. Send packet
        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, 
                                   in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    # ========== 3) Statistics Collection and Parsing ==========
    def request_stats(self):
        self.attack_count = 0
        self.benign_count = 0
        self.stats_event.clear()
        self.stats_replies_expected = len(self.datapaths)

        self.logger.info(">>> Starting flow table statistics request for all switches...")
        for dp in self.datapaths.values():
            dp.send_msg(dp.ofproto_parser.OFPFlowStatsRequest(dp))
        
        try:
            self.stats_event.wait(timeout=5)
        except hub.Timeout:
            self.logger.warning("Flow table statistics reply timed out!")
        self.logger.info("<<< Flow table statistics request completed.")

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        datapath = ev.msg.datapath
        parser = datapath.ofproto_parser
        is_deployed = dpid in self.deployed_ml_switches

        for stat in ev.msg.body:
            if 'ipv4_dst' in stat.match:
                flow_key = (dpid, tuple(sorted(stat.match.items())))
                pkts = stat.packet_count - self.flow_stats.get(flow_key, 0)
                self.flow_stats[flow_key] = stat.packet_count

                if pkts > 0:
                    dst_ip = stat.match.get('ipv4_dst')
                    src_ip = stat.match.get('ipv4_src') 
                    
                    # Attack detection logic based on iperf
                    is_attack = False
                    if (dst_ip == config.DEST_IP and 
                        'ip_proto' in stat.match and 
                        stat.match['ip_proto'] == 17):  # UDP
                        is_attack = True

                    if is_attack:
                        self.attack_count += pkts
                        src_ip = stat.match.get('ipv4_src', 'RANDOM')
                        
                        if is_deployed and random.random() < self.detection_rate:
                            self.logger.info(f"[Intercept] s{dpid}: Detected UDP attack flow {src_ip}->{dst_ip}, packets {pkts}.")
                            self.add_flow(datapath, 20, stat.match, [], 
                                         idle_timeout=config.EPISODE_LENGTH)
                    else:
                        self.benign_count += pkts
        
        self.stats_replies_expected -= 1
        if self.stats_replies_expected <= 0:
            self.stats_event.set()

    # ========== 4) Training Loop ==========
    def _training_initializer(self):
        """Wait for Mininet API readiness, then start the main training loop"""
        api_url = f"{config.MININET_API_URL}/health"
        while not self.training_started:
            try:
                r = requests.get(api_url, timeout=1)
                if r.status_code == 200:
                    self.logger.info("*** Mininet API is ready. Starting main training loop. ***")
                    self.training_started = True
                    hub.spawn(self._training_loop)
                    return
            except requests.exceptions.RequestException:
                self.logger.info("... Waiting for Mininet API, retrying in 2s ...")
            hub.sleep(2)

    def _clear_all_flows(self):
        self.logger.info("Clearing all flow table entries on all switches...")
        for datapath in self.datapaths.values():
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            mod = parser.OFPFlowMod(datapath=datapath, command=ofproto.OFPFC_DELETE,
                                    out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
                                    match=parser.OFPMatch())
            datapath.send_msg(mod)
            self._add_default_flow(datapath)
        hub.sleep(1)

    def _training_loop(self):
        """Main training loop - supports PPO and GNN+PPO"""
        self._clear_all_flows()
        num_switches = len(self.datapaths)
        num_hosts = len(self.hosts)
        
        try:
            # Initialize environment
            self.env = SDNEnvironment(
                self, num_switches, num_hosts, 
                use_gnn=self.use_gnn_ppo
            )
            
            # Select agent based on configuration
            if self.use_gnn_ppo:
                from gnn_ppo_agent import GNNPPOAgent
                self.agent = GNNPPOAgent(
                    node_feature_dim=config.NODE_FEATURE_DIM,
                    hidden_dim=config.GNN_HIDDEN_DIM,
                    num_gnn_layers=config.GNN_NUM_LAYERS,
                    gnn_type=config.GNN_TYPE,
                    lr=config.PPO_LR,
                    gamma=config.PPO_GAMMA,
                    eps_clip=config.PPO_EPS_CLIP,
                    k_epochs=config.PPO_K_EPOCHS,
                    gae_lambda=config.PPO_GAE_LAMBDA,
                    entropy_coef=config.PPO_ENTROPY_COEF,
                    value_loss_coef=config.PPO_VALUE_LOSS_COEF,
                    max_grad_norm=config.PPO_MAX_GRAD_NORM
                )
                self.logger.info(f"Initialized GNN+PPO agent: "
                                f"GNN type={config.GNN_TYPE}, "
                                f"Hidden dimension={config.GNN_HIDDEN_DIM}, "
                                f"GNN layers={config.GNN_NUM_LAYERS}")
            else:
                from ppo_agent import PPOAgent
                self.agent = PPOAgent(self.env.state_dim, self.env.action_dim)
                self.logger.info("Initialized traditional PPO agent")
                
        except Exception as e:
            self.logger.critical(f"Failed to initialize RL components: {e}", exc_info=True)
            return

        update_interval = getattr(config, 'PPO_UPDATE_INTERVAL', 10)
        save_interval = getattr(config, 'MODEL_SAVE_INTERVAL', 500)
        
        for episode in range(config.NUM_EPISODES):
            try:
                self.flow_stats.clear()
                state = self.env.reset()
                
                # Get action
                action, log_prob, value = self.agent.get_action(state)
                
                # Execute action
                next_state, reward, done = self.env.step(action)
                
                # Store experience
                self.agent.store_transition(state, action, log_prob, reward, value, done)
                
                # Periodic update
                if (episode + 1) % update_interval == 0:
                    self.agent.update()
                    avg_reward = self.env.get_average_reward(last_n=update_interval)
                    self.logger.info(f"Update complete | Average reward over last {update_interval} episodes: {avg_reward:.4f}")
                
                # Logging
                self.logger.info(f"Episode {episode + 1}/{config.NUM_EPISODES}: "
                               f"Reward={reward:.4f}, "
                               f"Attack packets={self.attack_count}, "
                               f"Normal packets={self.benign_count}, "
                               f"Deployed={len(self.deployed_ml_switches)} switches")
                
                # Save model
                if (episode + 1) % save_interval == 0:
                    self.agent.save_models(episode + 1)
                    
            except Exception as e:
                self.logger.error(f"Error in training episode {episode + 1}: {e}", exc_info=True)
            hub.sleep(1)
        
        # Training complete, save final model
        self.agent.save_models(config.NUM_EPISODES)
        self.logger.info("Training completed!")

    # ========== 5) Utility Methods ==========
    def add_flow(self, datapath, priority, match, actions, idle_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                instructions=inst, idle_timeout=idle_timeout)
        datapath.send_msg(mod)


# ========== REST API ==========
class SdnRestApi(ControllerBase):
    """SDN Controller REST API"""
    
    def __init__(self, req, link, data, **config):
        super(SdnRestApi, self).__init__(req, link, data, **config)
        self.controller = data[SDN_CONTROLLER_INSTANCE_NAME]

    @route('sdn', '/ready', methods=['GET'])
    def is_ready(self, req, **kwargs):
        body = {'ready': self.controller.topology_ready}
        return Response(content_type='application/json', json=body)
    
    @route('sdn', '/stats', methods=['GET'])
    def get_stats(self, req, **kwargs):
        """Retrieve training statistics."""
        stats = {
            'topology_ready': self.controller.topology_ready,
            'training_started': self.controller.training_started,
            'deployed_switches': list(self.controller.deployed_ml_switches),
            'attack_count': self.controller.attack_count,
            'benign_count': self.controller.benign_count,
            'use_gnn_ppo': self.controller.use_gnn_ppo
        }
        return Response(content_type='application/json', json=stats)
