import os
import json
import argparse
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

from tqdm import tqdm

# import action_type as actiontype

def save_image(
    example,
    image_height,
    image_width,
    image_channels,
    path,
):
    """Decodes image from example and reshapes.

    Args:
        example: Example which contains encoded image.
        image_height: The height of the raw image.
        image_width: The width of the raw image.
        image_channels: The number of channels in the raw image.

    Returns:
        Decoded and reshaped image tensor.
    """
    image = tf.io.decode_raw(
        example.features.feature['image/encoded'].bytes_list.value[0],
        out_type=tf.uint8,
    )

    height = tf.cast(image_height, tf.int32)
    width = tf.cast(image_width, tf.int32)
    n_channels = tf.cast(image_channels, tf.int32)

    image = tf.reshape(image, (height, width, n_channels))

    # 编码回图片
    image = tf.image.encode_png(image)
    # 保存
    with tf.io.gfile.GFile(path, 'wb') as file:
        file.write(image.numpy())

def save_png(img, path):
    with open(path, "wb") as output:
        output.write(img)

def save_bin(byte, path):
    with open(path,'wb') as file:
        file.write(byte)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AndroidControl')
args = parser.parse_args()

dataset_name = args.dataset #'AndroidControl'

dataset_directories = {
    'AndroidControl': '/data/true_nas/zfs_share1/zyc/data/data/google_research/android_control/*', #android_control-00000-of-00020',
}

filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
failed_files = []

for filename in filenames:
    try:
        # Process each file individually to isolate failures
        dataset = tf.data.TFRecordDataset(filename, compression_type='GZIP').as_numpy_iterator()
        
        for d in tqdm(dataset, desc=f'Processing {os.path.basename(filename)}'):
            try:
                ex = tf.train.Example()
                ex.ParseFromString(d)
                episode_id = ex.features.feature['episode_id'].int64_list.value[0]
                save_path = os.path.join('/data/true_nas/zfs_share1/zyc/data/data/google_research/android_control_extracted/translated_ui', 'AndroidControl', f'episode_{episode_id}')
                if os.path.exists(save_path):
                    print(f'episode {episode_id} already exists')
                    continue
                os.makedirs(save_path, exist_ok=True)
                json_path = os.path.join(save_path, f'episode_{episode_id}.json')
                json_data = {}
                goal = ex.features.feature['goal'].bytes_list.value[0].decode('utf-8')
                json_data['episode_id'] = episode_id
                json_data['goal'] = goal
                screenshots = ex.features.feature['screenshots'].bytes_list.value
                screenshots_len = len(screenshots)
                screenshot_path = {}
                for idx in range(screenshots_len):
                    img = screenshots[idx]
                    img_path = os.path.join(save_path, f'{episode_id}_{idx}.png')
                    save_png(img, img_path)
                    screenshot_path[idx] = img_path
                json_data['screenshots'] = screenshot_path
                acc_trees = ex.features.feature['accessibility_trees'].bytes_list.value
                acc_tree_path = {}
                for idx in range(len(acc_trees)):
                    acc_tree = acc_trees[idx]
                    acc_path = os.path.join(save_path, f'{episode_id}_{idx}_AccTree.bin')
                    save_bin(acc_tree, acc_path)
                    acc_tree_path[idx] = acc_path
                json_data['accessibility_trees'] = acc_tree_path
                actions = ex.features.feature['actions'].bytes_list.value
                step_instructions = ex.features.feature['step_instructions'].bytes_list.value
                actions_len = len(actions)
                json_data['steps'] = []
                for step_idx in range(actions_len):
                    action = actions[step_idx].decode('utf-8')
                    json_action = json.loads(action)
                    width = ex.features.feature['screenshot_widths'].int64_list.value[step_idx]
                    height = ex.features.feature['screenshot_heights'].int64_list.value[step_idx]
                    step_instruction = step_instructions[step_idx].decode('utf-8')
                    step = {
                        'step_id': step_idx,
                        'image_before': screenshot_path[step_idx],
                        'image_after': screenshot_path[step_idx+1] if step_idx+1 < screenshots_len else None,
                        'image_size':[width, height],
                        'action': json_action,
                        'step_instruction': step_instruction,
                    }
                    json_data['steps'].append(step)
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Error processing episode {episode_id}: {str(e)}")
                continue
                
    except tf.errors.DataLossError as e:
        print(f"Failed to read file {filename}: Corrupted GZIP data")
        failed_files.append(filename)
        continue
    except Exception as e:
        print(f"Unexpected error reading file {filename}: {str(e)}")
        failed_files.append(filename)
        continue

if failed_files:
    print("\nThe following files failed to process:")
    for f in failed_files:
        print(f"- {f}")