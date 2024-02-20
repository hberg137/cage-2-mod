import py_trees_devel.py_trees as py_trees
import numpy as np
import copy
from Agents.PPOAgent import PPOAgent
import torch
import torch.nn as nn
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GetAction(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "GetAction"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action_space", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)

        #self.agent = agent

    def update(self):
        if self.blackboard.step >= 3:
            self.blackboard.agent.add_scan(self.blackboard.observation)
            self.blackboard.action = self.blackboard.agent.agent.get_action(self.blackboard.observation)
            # print("in action")
        return py_trees.common.Status.SUCCESS


class SetupCheck(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Setup?"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)
    
    def update(self):
        if self.blackboard.step < 3:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class Setup(py_trees.behaviour.Behaviour):

    def __init__(self, agent, name: str = "Setup Action"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "decoy_ids", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action_space", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "scan_state", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "start_actions", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)

        self.blackboard.agent = agent

        self.blackboard.decoy_ids = list(range(1000, 1009))
        self.blackboard.action_space = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22, 11, 12, 13, 14,
                                        141, 142, 143, 144, 132, 2, 15, 24, 25, 26, 27] + self.blackboard.decoy_ids
        #print(self.blackboard.action_space)
        self.blackboard.scan_state = np.zeros(10)
        self.blackboard.start_actions = [51, 116, 55]

    def update(self):

        # scan_state_copy = copy.copy(self.blackboard.agent.scan_state)
        
        self.blackboard.agent.add_scan(self.blackboard.observation)

        # print(self.blackboard.agent.start_actions)

        if len(self.blackboard.agent.start_actions) > 0:
            #PPOAgent.add_scan(self.blackboard.agent, self.blackboard.observation)
            # super(type(self.blackboard.agent), self.blackboard.agent).add_scan(self.blackboard.observation)
            self.blackboard.action = self.blackboard.agent.start_actions[0]
            self.blackboard.agent.start_actions = self.blackboard.agent.start_actions[1:]
            # print(self.blackboard.start_actions)
            # print(len(self.blackboard.agent.start_actions))

        # print(self.blackboard.observation)
        return py_trees.common.Status.SUCCESS


class ChangeStratCheck(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Change Strategy?"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)
        
        self.blackboard.register_key(key = "switch", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "window", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "lstm_model", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "need_switch", access = py_trees.common.Access.WRITE)
    
    def update(self):
        if self.blackboard.step == 3 or self.blackboard.step == self.blackboard.switch.switch_step:
            return py_trees.common.Status.SUCCESS
        # elif self.blackboard.need_switch and len(self.blackboard.window) == 5:
        #     tens_window = torch.tensor(self.blackboard.window, dtype=torch.float32).unsqueeze(0).to(device)
        #     out = self.blackboard.lstm_model(tens_window)
        #     if (out > 0.5).item():
        #         self.blackboard.need_switch = False
        #         return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class ChangeStrat(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Change Strategy"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "scan_state", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)

    def update(self):
        if self.blackboard.step == 3:
            scan_state_copy = copy.copy(self.blackboard.agent.scan_state)
            # print(self.blackboard.agent.scan_state)
            
            self.blackboard.agent.add_scan(self.blackboard.observation)

            if self.blackboard.agent.fingerprint_meander():
                self.blackboard.agent.agent = self.blackboard.agent.load_meander()
            elif self.blackboard.agent.fingerprint_bline():
                self.blackboard.agent.agent = self.blackboard.agent.load_bline()
            else:
                self.blackboard.agent.agent = self.blackboard.agent.load_sleep()
        
            #print(self.blackboard.agent.agent)
            # add decoys and scan state
            self.blackboard.agent.agent.current_decoys = {1000: [55], # enterprise0
                                                    1001: [], # enterprise1
                                                    1002: [], # enterprise2
                                                    1003: [], # user1
                                                    1004: [51, 116], # user2
                                                    1005: [], # user3
                                                    1006: [], # user4
                                                    1007: [], # defender
                                                    1008: []} # opserver0
            # add old since it will add new scan in its own action (since recieves latest observation)
            self.blackboard.agent.agent.scan_state = scan_state_copy
            self.blackboard.agent.agent_loaded = True
        else:
            self.blackboard.agent.agent = self.blackboard.agent.load_bline()

        return py_trees.common.Status.SUCCESS


class GetPPOAction(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Get PPO Action"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action_space", access = py_trees.common.Access.WRITE)


    def update(self):
        
        #self.blackboard.action = self.agent.get_action(self.blackboard.observation)

        self.blackboard.agent.agent.add_scan(self.blackboard.observation)
        self.blackboard.observation = self.blackboard.agent.agent.pad_observation(self.blackboard.observation)
        state = torch.FloatTensor(self.blackboard.observation.reshape(1, -1)).to(device)
        action = self.blackboard.agent.agent.old_policy.act(state, self.blackboard.agent.agent.memory,
                                                            deterministic = self.blackboard.agent.agent.deterministic)
        self.blackboard.action = self.blackboard.action_space[action]
        
        
        return py_trees.common.Status.SUCCESS


class DeployDecoyCheck(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Action is Deploy Decoy?"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "decoy_ids", access = py_trees.common.Access.WRITE)

    def update(self):
        if self.blackboard.action in self.blackboard.decoy_ids:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class DeployDecoy(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Deploy Decoy"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "decoy_ids", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action_space", access = py_trees.common.Access.WRITE)

    def update(self):
        host = self.blackboard.action
        try:
        # pick the top remaining decoy
            self.blackboard.action = [a for a in self.blackboard.agent.agent.greedy_decoys[host]
                                        if a not in self.blackboard.agent.agent.current_decoys[host]][0]
            self.blackboard.agent.agent.add_decoy(self.blackboard.action, host)
        except:
            state = torch.FloatTensor(self.blackboard.observation.reshape(1, -1)).to(device)
            actions = self.blackboard.agent.agent.old_policy.act(state, self.blackboard.agent.agent.memory, full=True)
            max_actions = torch.sort(actions, dim=1, descending=True)
            max_actions = max_actions.indices
            max_actions = max_actions.tolist()

            # don't need top action since already know it can't be used (hence could put [1:] here, left for clarity)
            for action_ in max_actions[0]:
                a = self.blackboard.action_space[action_]
                # if next best action is decoy, check if its full also
                if a in self.blackboard.agent.agent.current_decoys.keys():
                    if len(self.blackboard.agent.agent.current_decoys[a]) < len(self.blackboard.agent.agent.greedy_decoys[a]):
                        self.blackboard.action = self.blackboard.agent.agent.select_decoy(a, self.blackboard.observation)
                        self.blackboard.agent.agent.add_decoy(self.blackboard.action, a)
                        break
                else:
                    # don't select a next best action if "restore", likely too aggressive for 30-50 episodes
                    if a not in self.blackboard.agent.agent.restore_decoy_mapping.keys():
                        self.blackboard.action = a
                        break
        
        return py_trees.common.Status.SUCCESS


class RemoveDecoysCheck(py_trees.behaviour.Behaviour):
    
    def __init__(self, name: str = "Action is Restore Host?"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)

    def update(self):
        if self.blackboard.action in self.blackboard.agent.agent.restore_decoy_mapping.keys():
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class RemoveDecoys(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Remove Decoys"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "decoy_ids", access = py_trees.common.Access.WRITE)

    def update(self):
        for decoy in self.blackboard.agent.agent.restore_decoy_mapping[self.blackboard.action]:
            for host in self.blackboard.decoy_ids:
                if decoy in self.blackboard.agent.agent.current_decoys[host]:
                    self.blackboard.agent.agent.current_decoys[host].remove(decoy)

        return py_trees.common.Status.SUCCESS


class ExecuteActions(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Execute Actions"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "wrapped_cyborg", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "r", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "a", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "cyborg", access = py_trees.common.Access.WRITE)

        self.blackboard.register_key(key = "test_counter", access = py_trees.common.Access.WRITE)

        self.blackboard.register_key(key = "states", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "labels", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "switch", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "window", access = py_trees.common.Access.WRITE)

    def update(self):
        #print(self.blackboard.observation)
        self.blackboard.observation, reward, done, info = self.blackboard.wrapped_cyborg.step(self.blackboard.action)
        self.blackboard.r.append(reward)
        self.blackboard.states.append(self.blackboard.observation)
        self.blackboard.window.append(self.blackboard.observation)
        if len(self.blackboard.window) > 5:
            self.blackboard.window.pop(0)
        if self.blackboard.step < self.blackboard.switch.switch_step:
            self.blackboard.labels.append([0])
        else:
            self.blackboard.labels.append([1])
        self.blackboard.a.append((str(self.blackboard.cyborg.get_last_action('Blue')),
                                  str(self.blackboard.cyborg.get_last_action('Red'))))
        
        self.blackboard.test_counter += 1
        #print(self.blackboard.test_counter)

        return py_trees.common.Status.SUCCESS