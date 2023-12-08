import random

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared.Actions import PrivilegeEscalate, ExploitRemoteService, DiscoverRemoteSystems, Impact, \
    DiscoverNetworkServices, Sleep

class RedAgent(BaseAgent):

    def __init__(self, agent_type: str = "Meander"):
        self.agent_type = agent_type
        
        # Meander agent
        #if agent_type == 'Meander':      
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None
        # B line agent
        #elif agent_type == 'B_line':
        self.action = 0
        self.target_ip_address = None
        self.last_subnet = None
        self.last_ip_address = None
        self.action_history = {}
        self.jumps = [0,1,2,2,2,2,5,5,5,5,9,9,9,12,13]
        self.start_bline = False

    def get_action(self, observation, action_space):
        if self.agent_type == 'Meander':
            return self.get_action_meander(observation, action_space)
        elif self.agent_type == 'B_line':
            return self.get_action_bline(observation, action_space)
        else:
            return Sleep()

    def get_action_meander(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        self._process_success(observation)
        print("in meander")

        session = list(action_space['session'].keys())[0]

        # Always impact if able
        if 'Op_Server0' in self.escalated_hosts:
            self.last_host = 'Op_Server0'
            return Impact(agent='Red', hostname='Op_Server0', session=session)

        # start by scanning
        #print(observation)
        for subnet in action_space["subnet"]:
            if not action_space["subnet"][subnet] or subnet in self.scanned_subnets:
                continue
            self.scanned_subnets.append(subnet)

            # self.action = 0
            action = DiscoverRemoteSystems(subnet=subnet, agent='Red', session=session)
            # if self.action not in self.action_history:
            #     self.action_history[self.action] = action
            return action
        
        # discover network services
        # # act on ip addresses discovered in first subnet
        addresses = [i for i in action_space["ip_address"]]
        random.shuffle(addresses)
        for address in addresses:
            if not action_space["ip_address"][address] or address in self.scanned_ips:
                continue
            self.scanned_ips.append(address)
            #print(observation)

            # enterprise = [x for x in observation if 'Enterprise' in x]
            # op_server = [x for x in observation if 'Op_Server' in x]
            
            # if self.action > 3:
            #     try:
            #         [value for key, value in observation.items() if key != 'success'][2]['Interface'][0]['IP Address']
            #         print("yeah?")
            #     except:
            #         pass

            # if len(enterprise) > 0:
            #     self.action = 4
            # else:
            #     if len(op_server) > 0:
            #         self.action = 11
            #     self.action = 1
            action = DiscoverNetworkServices(ip_address=address, agent='Red', session=session)
            # if self.action not in self.action_history:
            #     self.action_history[self.action] = action
            return action
        # priv esc on owned hosts
        hostnames = [x for x in action_space['hostname'].keys()]
        random.shuffle(hostnames)
        for hostname in hostnames:
            # test if host is not known
            if not action_space["hostname"][hostname]:
                continue
            # test if host is already priv esc
            if hostname in self.escalated_hosts:
                continue
            # test if host is exploited
            if hostname in self.host_ip_map and self.host_ip_map[hostname] not in self.exploited_ips:
                continue
            self.escalated_hosts.append(hostname)
            self.last_host = hostname
            return PrivilegeEscalate(hostname=hostname, agent='Red', session=session)

        # access unexploited hosts
        for address in addresses:
            # test if output of observation matches expected output
            if not action_space["ip_address"][address] or address in self.exploited_ips:
                continue
            self.exploited_ips.append(address)
            self.last_ip = address
            return ExploitRemoteService(ip_address=address, agent='Red', session=session)

        raise NotImplementedError('Red Meander has run out of options!')
    
    def get_action_bline(self, observation, action_space):
        # print(self.action)
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        session = 0
        print("in b line")

        while True:
            if (observation['success'] == True) and (self.start_bline):
                self.action += 1 if self.action < 14 else 0
            else:
                self.action = self.jumps[self.action]
                self.start_bline = True

            if self.action in self.action_history:
                action = self.action_history[self.action]

            # Discover Remote Systems
            elif self.action == 0:
                # print("p")
                # print(observation)
                self.initial_ip = observation['User0']['Interface'][0]['IP Address']
                self.last_subnet = observation['User0']['Interface'][0]['Subnet']
                action = DiscoverRemoteSystems(session=session, agent='Red', subnet=self.last_subnet)
            # Discover Network Services- new IP address found
            elif self.action == 1:
                hosts = [value for key, value in observation.items() if key != 'success']
                get_ip = lambda x : x['Interface'][0]['IP Address']
                interfaces = [get_ip(x) for x in hosts if get_ip(x)!= self.initial_ip]
                self.last_ip_address = random.choice(interfaces)
                action =DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)

            # Exploit User1
            elif self.action == 2:
                 action = ExploitRemoteService(session=session, agent='Red', ip_address=self.last_ip_address)

            # Privilege escalation on User Host
            elif self.action == 3:
                hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)

            # Discover Network Services- new IP address found
            elif self.action == 4:
                self.enterprise_host = [x for x in observation if 'Enterprise' in x][0]
                self.last_ip_address = observation[self.enterprise_host]['Interface'][0]['IP Address']
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)

            # Exploit- Enterprise Host
            elif self.action == 5:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
                action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)

            # Privilege escalation on Enterprise Host
            elif self.action == 6:
                hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)

            # Scanning the new subnet found.
            elif self.action == 7:
                self.last_subnet = observation[self.enterprise_host]['Interface'][0]['Subnet']
                action = DiscoverRemoteSystems(subnet=self.last_subnet, agent='Red', session=session)

            # Discover Network Services- Enterprise2
            elif self.action == 8:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][2]['Interface'][0]['IP Address']
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.target_ip_address)

            # Exploit- Enterprise2
            elif self.action == 9:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
                action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)

            # Privilege escalation on Enterprise2
            elif self.action == 10:
                hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)

            # Discover Network Services- Op_Server0
            elif self.action == 11:
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=observation['Op_Server0']['Interface'][0]['IP Address'])

            # Exploit- Op_Server0
            elif self.action == 12:
                info = [value for key, value in observation.items() if key != 'success']
                if len(info) > 0:
                    action = ExploitRemoteService(agent='Red', session=session, ip_address=info[0]['Interface'][0]['IP Address'])
                else:
                    self.action = 0
                    continue
            # Privilege escalation on Op_Server0
            elif self.action == 13:
                action = PrivilegeEscalate(agent='Red', hostname='Op_Server0', session=session)
            # Impact on Op_server0
            elif self.action == 14:
                action = Impact(agent='Red', session=session, hostname='Op_Server0')

            if self.action not in self.action_history:
                self.action_history[self.action] = action
            return action
        
    def _process_success(self, observation):
        if self.last_ip is not None:
            if observation['success'] == True:
                self.host_ip_map[[value['System info']['Hostname'] for key, value in observation.items()
                                  if key != 'success' and 'System info' in value
                                  and 'Hostname' in value['System info']][0]] = self.last_ip
            else:
                self._process_failed_ip()
            self.last_ip = None
        if self.last_host is not None:
            if observation['success'] == False:
                if self.last_host in self.escalated_hosts:
                    self.escalated_hosts.remove(self.last_host)
                if self.last_host in self.host_ip_map and self.host_ip_map[self.last_host] in self.exploited_ips:
                    self.exploited_ips.remove(self.host_ip_map[self.last_host])
            self.last_host = None

    def _process_failed_ip(self):
        self.exploited_ips.remove(self.last_ip)
        hosts_of_type = lambda y: [x for x in self.escalated_hosts if y in x]
        if len(hosts_of_type('Op')) > 0:
            for host in hosts_of_type('Op'):
                self.escalated_hosts.remove(host)
                ip = self.host_ip_map[host]
                self.exploited_ips.remove(ip)
        elif len(hosts_of_type('Ent')) > 0:
            for host in hosts_of_type('Ent'):
                self.escalated_hosts.remove(host)
                ip = self.host_ip_map[host]
                self.exploited_ips.remove(ip)


    def get_agent_type(self):
        return self.agent_type
    
    def set_agent_type(self, agent_type):
        self.agent_type = agent_type


    def end_episode(self):
        if self.agent_type == 'Meander':
            self.end_episode_meander()
        elif self.agent_type == 'B_line':
            self.end_episode_bline()

    def end_episode_meander(self):
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None

    def end_episode_bline(self):
        self.action = 0
        self.target_ip_address = None
        self.last_subnet = None
        self.last_ip_address = None
        self.action_history = {}

    def train(self, results):
        pass

    def set_initial_values(self, action_space, observation):
        pass