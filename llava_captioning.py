import json
import time
import copy
import warnings
import numpy as np
import torch

from pathlib import Path
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    max_frames_num = int(max_frames_num)

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = float(vr.get_avg_fps())
    video_time = total_frame_num / avg_fps

    stride = max(1, int(round(avg_fps / float(fps))))
    frame_idx = list(range(0, total_frame_num, stride))
    frame_time = [i / avg_fps for i in frame_idx]

    if len(frame_idx) > max_frames_num or force_sample:
        frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
        frame_time = [i / avg_fps for i in frame_idx]

    frame_time = ",".join([f"{t:.2f}s" for t in frame_time])
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_time, video_time


pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained,
    None,
    model_name,
    torch_dtype="bfloat16",
    device_map=device_map,
    attn_implementation="eager",
)
model.eval()

# Base directory containing video subdirectories
base_dir = Path("/home/dataset/video_eval")

levels = ["L1", "L2", "L3", "L4", "L5"]
durations = ["short", "medium", "long"]

# Output file for captions
output_file = "captions_llava_video_7B_qwen2.jsonl"

# Load already processed videos
processed_videos = set()
if Path(output_file).exists():
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    processed_videos.add(data.get("video_name"))
                except json.JSONDecodeError:
                    continue
    print(f"Found {len(processed_videos)} already processed videos. Skipping them...")

max_frames_num = 64
conv_template = "qwen_1_5"

for level in levels:
    for duration in durations:
        video_dir = base_dir / level / duration

        if not video_dir.exists():
            print(f"Warning: Directory {video_dir} does not exist. Skipping...")
            continue

        print(f"\nProcessing {level}/{duration} videos...")

        video_files = sorted(video_dir.glob("*.mp4"))
        print(f"Found {len(video_files)} videos in {level}/{duration}")

        for idx, video_path in enumerate(video_files, 1):
            video_name = video_path.relative_to(base_dir).as_posix()

            if video_name in processed_videos:
                print(f"\n[{idx}/{len(video_files)}] Skipping (already processed): {video_name}")
                continue

            print(f"\n[{idx}/{len(video_files)}] Processing: {video_name}")

            try:
                start = time.time()

                video, frame_time, video_time = load_video(
                    str(video_path),
                    max_frames_num,
                    1,
                    force_sample=True,
                )

                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
                video = [video]

                question = DEFAULT_IMAGE_TOKEN + "\nDescribe the video in detail."

                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt_question,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).to(device)

                with torch.inference_mode():
                    cont = model.generate(
                        input_ids,
                        images=video,
                        modalities=["video"],
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=4096,
                    )

                caption = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

                result = {
                    "video_name": video_name,
                    "caption": caption,
                }

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                elapsed_time = time.time() - start
                print(f"  Completed in {elapsed_time:.2f}s")
                print(f"  Caption length: {len(caption)} characters")

            except Exception as e:
                error_msg = str(e)
                print(f"  Error processing {video_name}: {error_msg}")
                continue

print(f"\nAll videos processed. Results saved to {output_file}")
