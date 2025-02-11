import os
import json
import argparse
import random
import time


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


def round_robin_call(data, engine_list, output_data_path):
    active_engine_list = engine_list.split(",")
    engine2last_failure = {}
    for engine in active_engine_list:
        engine2last_failure[engine] = {"timestamp": None}

    for d in data:
        prompt = d["prompt"]
        reasoning_content = ""
        content = ""
        engine = get_engine(active_engine_list, engine2last_failure)
        try:
            reasoning_content, content = engine2func[engine](prompt)
            d["reasoning_content"] = reasoning_content
            d["content"] = content
            d["engine"] = engine
            with open(output_data_path, "a") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

            engine2last_failure[engine]["timestamp"] = None
        except Exception as e:
            engine2last_failure[engine]["timestamp"] = time.time()
            print(f"Error: {e}")
            print(f"Failed to call {engine}.")


def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(line.strip())
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
    else:
        output_data = []
    already_called_ids = [item["id"] for item in output_data]
    print(f"already called {len(already_called_ids)} prompts")
    prompt_data = [item for item in prompt_data if item["id"] not in already_called_ids]
    print(f"remaining {len(prompt_data)} prompts")

    from multiprocessing import Process

    def process_chunk(prompts, engine_list, output_data_path):
        for prompt in prompts:
            round_robin_call(prompt, engine_list, output_data_path)

    # 将数据分成8份(可以根据需要调整进程数)
    num_processes = 2
    chunk_size = len(prompt_data) // num_processes
    chunks = [
        prompt_data[i : i + chunk_size] for i in range(0, len(prompt_data), chunk_size)
    ]

    # 创建并启动多个进程
    processes = []
    for chunk in chunks:
        p = Process(
            target=process_chunk, args=(chunk, args.engine_list, args.output_data_path)
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    print(f"finished")


if __name__ == "__main__":
    main()
