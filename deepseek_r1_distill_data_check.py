import json
import os
import argparse
import re
from tqdm import tqdm
from typing import Tuple
from ast import literal_eval

def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


def parse_parameter(parameter: str) -> dict:
    """
    Parser the parameter string into a dictionary.
    key is the name of the parameter, value is the value of the parameter.
    parameter is like:
    'point: [426, 270]' -> {'point': [426, 270]}
    'region: [20, 300, 400, 500]' -> {'region': [20, 300, 400, 500]}
    'direction: up' -> {'direction': 'up'}
    'text: Click to manage account information.' -> {'text': 'Click to manage account information.'}
    """
    try:
        if "region:" in parameter:
            parameter = parameter.strip().split(":", 1)
            key = parameter[0].strip()
            value = literal_eval(parameter[1].strip().split("]")[0]+"]")
            return {key: value}
        else:
            name = extract_xml(parameter, "param_name").strip()
            value = extract_xml(parameter, "param_value").strip()
            if name in ['point']:
                return {name: literal_eval(value.strip().split("]")[0]+"]")}
            else:
                return {name: value}
            
    except Exception as e:
        print(f"Error parsing parameter: {e}")
        print(f"Error Parsing Parameter: {parameter}")
        return {}


def extract_action(text: str) -> str:
    """
    Extract the ground truth action from the anwser.

    <answer>
        <description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </description>
        <action>
            <action_name>
                TAP
            </action_name>
            <parameters>
                <parameter>
                    <param_name>
                        point
                    </param_name>
                    <param_value>
                        [426, 270]
                    </param_value>
                </parameter>
            </parameters>
        </action>
        <active region>
            region: [20, 300, 400, 500]
        </active region>
    </answer>

    a well defined action contains: name, parameters, active_region
    """
    answer = extract_xml(text, "answer").strip()
    action = extract_xml(answer, "action").strip()
    action_name = extract_xml(action, "action_name").strip()
    parameter = parse_parameter(
        extract_xml(action, "parameter").strip()
    )
    active_region = parse_parameter(extract_xml(answer, "active region").strip()).get("region", None)
    return action_name, parameter, active_region


def point_in_region(point: Tuple[int, int], region: Tuple[int, int, int, int]) -> bool:
    """
    Check if the point is in the region.
    point [x,y]
    region [x1,y1,x2,y2]
    """
    return region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]


def eval_action(gt_action, gt_parameter, gt_active_region, action, parameter, action_confusion_matrix) -> float:
    """
    Evaluate the action is correct and return the reward.
    Action Space includes:
        TAP, SWIPE, TYPE, with parameters: point, direction, text
        TASK_COMPLETE, PRESS_ENTER, TASK_IMPOSSIBLE, PRESS_BACK, PRESS_HOME, WAIT, with no parameters
    """

    def lcs(s1, s2):
        """
        Calculate the longest common subsequence (LCS) of two strings.
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    reward = 0.0
    if gt_action not in action_confusion_matrix:
        action_confusion_matrix[gt_action] = {}
    if action not in action_confusion_matrix[gt_action]:
        action_confusion_matrix[gt_action][action] = 0
    action_confusion_matrix[gt_action][action] += 1

    
    if gt_action != action:
        return reward # 动作不一致，奖励为0
    
    if gt_action == "TAP":
        point = parameter.get("point", None)
        if (
            isinstance(point, list)
            and isinstance(gt_active_region, list)
            and point_in_region(point, gt_active_region)
        ):
            reward += 1
        # else:
        #     print(f"TAP: point: {point}, gt_active_region: {gt_active_region}")
    
    elif gt_action == "SWIPE":
        direction = parameter.get("direction", None)
        gt_direction = gt_parameter.get("direction", None)
        if direction is not None and direction in ["up", "down", "left", "right"] and direction == gt_direction:
            reward += 1
        # else:
        #     print(f"SWIPE: direction: {direction}, gt_direction: {gt_direction}")
    
    elif gt_action == "TYPE":
        text = parameter.get("text", None)
        gt_text = gt_parameter.get("text", None)
        if text is not None and lcs(text, gt_text) / min(len(text), len(gt_text)) > 0.5:
            reward += 1
        else:
            print(f"TYPE: text: {text}, gt_text: {gt_text}")

    elif gt_action == "TASK_COMPLETE":
        if action == "TASK_COMPLETE":
            reward += 1
    
    elif gt_action == "PRESS_ENTER":
        if action == "PRESS_ENTER":
            reward += 1
        
    elif gt_action == "TASK_IMPOSSIBLE":
        if action == "TASK_IMPOSSIBLE":
            reward += 1
        
    elif gt_action == "PRESS_BACK":
        if action == "PRESS_BACK":
            reward += 1
        
    elif gt_action == "PRESS_HOME":
        if action == "PRESS_HOME":
            reward += 1
        
    elif gt_action == "WAIT":
        if action == "WAIT":
            reward += 1
    else:
        reward += 0.0

    return reward

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def check_data(data):
    call_engine_count = {}
    unique_episodes = set()
    action_correction_count = {}
    action_confusion_matrix = {}
    action_space = [
        "TAP",
        "SWIPE",
        "TYPE",
        "TASK_COMPLETE",
        "PRESS_ENTER",
        "TASK_IMPOSSIBLE",
        "PRESS_BACK",
        "PRESS_HOME",
        "WAIT"
    ]
    for action in action_space:
        action_confusion_matrix[action] = {}
        for action_ in action_space:
            action_confusion_matrix[action][action_] = 0
    filtered_data = []

    for item in data:
        if item["engine"] not in call_engine_count:
            call_engine_count[item["engine"]] = 0
        call_engine_count[item["engine"]] += 1
        step_index = item["step_index"]
        unique_episodes.add(step_index.split("_")[0])
        gt_action, gt_parameter, gt_active_region = extract_action(item["gt_action"])
        action, parameter,_ = extract_action(item["content"])
        # print(f"gt_action: {gt_action}, action: {action}, parameter: {parameter}, gt_parameter: {gt_parameter}, gt_active_region: {gt_active_region}")
        reward = eval_action(gt_action, gt_parameter, gt_active_region, action, parameter, action_confusion_matrix)
        # print(f"reward: {reward}")
        if gt_action not in action_correction_count:
            action_correction_count[gt_action] = {'correct': 0, 'incorrect': 0}
        if reward > 0:
            filtered_data.append(item)
            action_correction_count[gt_action]['correct'] += 1
        else:
            action_correction_count[gt_action]['incorrect'] += 1
        
    print(f"call_engine_count: {call_engine_count}")
    print(f"unique_episodes count: {len(unique_episodes)}")
    for action, count in action_correction_count.items():
        print(f"{action}: accuracy: {count['correct']}/{count['correct'] + count['incorrect']}")
    print(f"action_confusion_matrix")
    for gt_action, action_count in action_confusion_matrix.items():
        print(f"{gt_action}:")
        for action, count in action_count.items():
            print(f"    {action}: {count}")
    return filtered_data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    data = load_data(args.data_path)
    print(f"loaded {len(data)} lines from {args.data_path}")

    filtered_data = check_data(data)
    with open(args.data_path.replace(".json", "_correct_filtered.json"), "w") as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()  