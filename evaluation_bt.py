import subprocess
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
import random

import py_trees_devel.py_trees as py_trees
import evaluation_bt_nodes as bt_nodes

import torch
from controller.ctrl import LSTMModel

import numpy as np

import random

MAX_EPS = 1000
agent_name = 'Blue'
random.seed(0)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Behavior tree

# Build blackboard
def build_blackboard():
    blackboard = py_trees.blackboard.Client(name = "Global")
    blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "action_space", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "cyborg", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "wrapped_cyborg", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
    #blackboard.register_key(key = "reward", access = py_trees.common.Access.WRITE)

    blackboard.register_key(key = "start_actions", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "scan_state", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "agent_loaded", access = py_trees.common.Access.WRITE)

    blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "r", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "a", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "test_counter", access = py_trees.common.Access.WRITE)
    
    blackboard.register_key(key = "states", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "labels", access = py_trees.common.Access.WRITE)
    
    blackboard.register_key(key = "switch", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "window", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "lstm_model", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "need_switch", access = py_trees.common.Access.WRITE)
    

    return blackboard

def build_bt(agent):
    root = py_trees.composites.Sequence(name = "CAGE Challenge BT", memory = True)

    determine_action = py_trees.composites.Selector(name = "Determine Action", memory = False)
    
    setup_seq = py_trees.composites.Sequence(name = "Setup Steps", memory = True)
    setup_check = bt_nodes.SetupCheck()
    setup = bt_nodes.Setup(agent)
    setup_seq.add_children([setup_check, setup])

    main_action_seq = py_trees.composites.Sequence(name = "Main Action Steps", memory = True)

    change_strat_sel = py_trees.composites.Selector(name = "Change Strategy Selector", memory = False)
    change_strat_check = bt_nodes.ChangeStratCheck()
    change_strat_check_inv = py_trees.decorators.Inverter(name = "Inverter", child = change_strat_check)
    change_strat = bt_nodes.ChangeStrat()
    change_strat_sel.add_children([change_strat_check_inv, change_strat])

    get_ppo_action = bt_nodes.GetPPOAction()

    deploy_decoy_sel = py_trees.composites.Selector(name = "Deploy Decoy Selector", memory = False)
    deploy_decoy_check = bt_nodes.DeployDecoyCheck()
    deploy_decoy_check_inv = py_trees.decorators.Inverter(name = "Inverter", child = deploy_decoy_check)
    deploy_decoy = bt_nodes.DeployDecoy()
    deploy_decoy_sel.add_children([deploy_decoy_check_inv, deploy_decoy])

    remove_decoys_sel = py_trees.composites.Selector(name = "Remove Decoys Selector", memory = False)
    remove_decoys_check = bt_nodes.RemoveDecoysCheck()
    remove_decoys_check_inv = py_trees.decorators.Inverter(name = "Inverter", child = remove_decoys_check)
    remove_decoys = bt_nodes.RemoveDecoys()
    remove_decoys_sel.add_children([remove_decoys_check_inv, remove_decoys])

    main_action_seq.add_children([change_strat_sel, get_ppo_action, deploy_decoy_sel, remove_decoys_sel])

    determine_action.add_children([setup_seq, main_action_seq])

    execute_actions = bt_nodes.ExecuteActions()

    root.add_children([determine_action, execute_actions])

    # setup = bt_nodes.Setup(agent)
    # get_action = bt_nodes.GetAction()
    # take_action = bt_nodes.TakeAction()
    # root.add_children([setup, get_action, take_action])

    #py_trees.display.render_dot_tree(root)
    return root

class StratSwitch:
    def __init__(self, switch_step) -> None:
        self.switch_step = switch_step

# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    random.seed(42)
    save_file_name = "results/" + scenario + "_RED_MB_MODEL_MB_FULL"
    
    min_sw_step = 10
    max_sw_step = 30
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'
    switch = StratSwitch(switch_step=random.randint(min_sw_step, max_sw_step))

    agent = MainAgent()
    
    # Create LSTM Model
    INPUT_DIM = 52
    HIDDEN_DIM = 100
    LAYER_DIM = 2
    OUT_DIM = 1
    LEARNING_RATE = 1e-3

    lstm_model = LSTMModel(INPUT_DIM, HIDDEN_DIM, LAYER_DIM, OUT_DIM).to(device)
    MODEL_PATH = 'Models/controller/lstm_model.pth'
    lstm_model.load_state_dict(torch.load(MODEL_PATH))
    lstm_model.eval()
    
    rewards_list = []
    # Change this line to load your agentobservation
    for num_steps in [100]:
        # Create behavior tree 
        blackboard = build_blackboard()
        
        blackboard.switch = switch
        blackboard.lstm_model = lstm_model
        
        blackboard.states = []
        blackboard.labels = []

        blackboard.agent = agent
        red_agent = RedMeanderAgent
        red2 = B_lineAgent
        blackboard.cyborg = CybORG(path, 'sim', agents={'Red': red_agent, 'Red2': red2}, strat_switch=switch)
        blackboard.wrapped_cyborg = wrap(blackboard.cyborg)

        blackboard.observation = blackboard.wrapped_cyborg.reset()
        # blackboard.states.append(blackboard.observation)
        # blackboard.labels.append([0])
        # observation = cyborg.reset().observation

        blackboard.action_space = blackboard.wrapped_cyborg.get_action_space(agent_name)

        
        # action_space = cyborg.get_action_space(agent_name)
        total_reward = []
        actions = []

        for i in range(MAX_EPS):
            print("EPISODE",i)
            blackboard.r = []
            blackboard.a = []
            blackboard.window = []
            blackboard.need_switch = True

            root = build_bt(agent)

            # get_action.setup()  # initialize the parameters for episode
            # cyborg.env.env.tracker.render()
            blackboard.switch.switch_step = 10

            blackboard.test_counter = 0
            blackboard.step = 0

            # subtract 3 because of setup steps
            for j in range(num_steps):
                root.tick_once()
                blackboard.step += 1

                #print(blackboard.cyborg.get_last_action("Red"))

                # result = cyborg.step(agent_name, action)
                
            agent.end_episode()
            rewards_list.append(blackboard.r)
            total_reward.append(sum(blackboard.r))
            actions.append(blackboard.a)
            # observation = cyborg.reset().observation
            blackboard.observation = blackboard.wrapped_cyborg.reset()
            print("ep done. reward is: ", sum(blackboard.r))
            switch.switch_step = random.randint(min_sw_step, max_sw_step)
    
    np.save(save_file_name + ".npy", np.array(rewards_list))
    # np.save("data/states.npy", np.array(blackboard.states))
    # np.save("data/labels.npy", np.array(blackboard.labels))
