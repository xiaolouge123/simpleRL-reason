import os
import json
import argparse
import random
import time
import signal  # new import to allow terminating the process group
from dotenv import load_dotenv

load_dotenv()


def get_deepseek_r1_response_deepseek(prompt, stream=False):
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="deepseek-reasoner", messages=messages, stream=stream
    )

    reasoning_content = ""
    content = ""

    if not stream:
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
    else:
        for chunk in response:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    return reasoning_content, content


def get_deepseek_r1_response_volcengine(prompt, stream=False):
    from openai import OpenAI

    api_key = os.environ.get("VOLCENGINE_API_KEY")
    client = OpenAI(
        api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3"
    )
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="ep-20250205221026-wtfnn", messages=messages, stream=stream
    )

    reasoning_content = ""
    content = ""

    if not stream:
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
    else:
        for chunk in response:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    return reasoning_content, content


def get_deepseek_r1_response_siliconflow(prompt, stream=False):
    from openai import OpenAI

    api_key = os.environ.get("SILICONFLOW_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1/")
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1", messages=messages, stream=stream
    )
    reasoning_content = ""
    content = ""

    if not stream:
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
    else:
        for chunk in response:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    return reasoning_content, content


def get_deepseek_r1_response_colossalai(prompt, stream=False):
    from openai import OpenAI

    api_key = os.environ.get("COLOSSALAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://cloud.luchentech.com/api/maas/")
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="VIP/deepseek-ai/DeepSeek-R1",
        messages=messages,
        stream=stream,
        max_tokens=4096,
    )
    reasoning_content = ""
    content = ""

    if not stream:
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
    else:
        for chunk in response:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    return reasoning_content, content


engine2func = {
    "deepseek": get_deepseek_r1_response_deepseek,
    "volcengine": get_deepseek_r1_response_volcengine,
    "siliconflow": get_deepseek_r1_response_siliconflow,
    "colossalai": get_deepseek_r1_response_colossalai,
}


def get_engine(active_engine_list, engine2last_failure):
    cnt = 0
    while True:
        engine_name = random.choice(active_engine_list)
        engine_status = engine2last_failure[engine_name]
        if (
            engine_status["timestamp"] is None
            or time.time() - engine_status["timestamp"] > 120
        ):
            return engine_name
        cnt += 1
        if cnt > 10:
            print("All engines are busy, waiting for 60 seconds...")
            time.sleep(60)
            cnt = 0


def round_robin_call(prompt_dict, engine_list, output_data_path, failure_counter, failure_lock):
    """
    This function calls one of the engine functions.
    On success, it resets the global failure counter.
    On failure, it increments the counter and if it reaches 100, exits the program.
    """
    active_engine_list = engine_list.split(",")
    engine2last_failure = {}
    for engine in active_engine_list:
        engine2last_failure[engine] = {"timestamp": None}

    # print("data content: ", prompt_dict)
    prompt = prompt_dict["prompt"]
    reasoning_content = ""
    content = ""
    engine = get_engine(active_engine_list, engine2last_failure)
    try:
        reasoning_content, content = engine2func[engine](prompt)
        prompt_dict["reasoning_content"] = reasoning_content
        prompt_dict["content"] = content
        prompt_dict["engine"] = engine
        with open(output_data_path, "a") as f:
            f.write(json.dumps(prompt_dict, ensure_ascii=False) + "\n")
        # On a successful call, reset the global failure counter
        with failure_lock:
            failure_counter.value = 0
        engine2last_failure[engine]["timestamp"] = None
    except Exception as e:
        with failure_lock:
            failure_counter.value += 1
            current_failures = failure_counter.value
        print(f"Error: {e}")
        print(f"Failed to call {engine}. Consecutive failures: {current_failures}")
        # If failure count reaches 100, terminate all processes in the current process group.
        with failure_lock:
            if failure_counter.value >= 100:
                print("Engine call failed 100 times in sequence, exiting program.")
                os.killpg(os.getpgrp(), signal.SIGTERM)
        engine2last_failure[engine]["timestamp"] = time.time()


def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_data_path", type=str, default=None)
    parser.add_argument("--output_data_path", type=str, default=None)
    parser.add_argument(
        "--engine_list", type=str, default="deepseek,volcengine,siliconflow,colossalai"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prompt_data = load_data(args.prompt_data_path)
    if os.path.exists(args.output_data_path):
        output_data = load_data(args.output_data_path)
        already_called_ids = [item["step_index"] for item in output_data]
    else:
        output_data = []
        already_called_ids = []
    print(f"already called {len(already_called_ids)} prompts")
    prompt_data = [item for item in prompt_data if item["step_index"] not in already_called_ids]
    random.shuffle(prompt_data)
    print(f"remaining {len(prompt_data)} prompts")
    # print("prompt_data content: ", prompt_data[0])
    
    from multiprocessing import Process, Value, Lock

    # Create a global shared failure counter and lock.
    failure_counter = Value('i', 0)
    failure_lock = Lock()

    def process_chunk(prompts, engine_list, output_data_path, failure_counter, failure_lock):
        """
        Process a chunk of prompt tasks.
        Before each call, we check the global failure counter.
        A progress bar is displayed for the prompts in this chunk.
        """
        from tqdm import tqdm
        for prompt_dict in tqdm(prompts, desc="Processing prompts", leave=True):
            with failure_lock:
                if failure_counter.value >= 100:
                    print("Exiting due to 100 consecutive failures.")
                    os.killpg(os.getpgrp(), signal.SIGTERM)
            round_robin_call(prompt_dict, engine_list, output_data_path, failure_counter, failure_lock)

    # Split the data into chunks (adjust number of processes as needed)
    num_processes = 10
    chunk_size = len(prompt_data) // num_processes
    chunks = [
        prompt_data[i : i + chunk_size] for i in range(0, len(prompt_data), chunk_size)
    ]

    processes = []
    for chunk in chunks:
        p = Process(
            target=process_chunk,
            args=(chunk, args.engine_list, args.output_data_path, failure_counter, failure_lock)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print(f"finished")


if __name__ == "__main__":
    main()
