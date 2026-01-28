# LLaVA-NeXT Video Captioning Tool 
* (LLaVA-Video-7B-Qwen2)

## 1. Environment Setup
- Clone this repository and navigate to the LLaVA folder
  ```bash
  git clone https://github.com/LLaVA-VL/LLaVA-NeXT
  cd LLaVA-NeXT
  ```
- Create and activate the Conda environment
  ```bash
  conda create -n llava python=3.10 -y
  conda activate llava
  ```
- Install the inference package
  ```bash
  pip install --upgrade pip
  pip install -e ".[train]"
  ```
## 2. Run Captioning
- Script: llava_captioning.py
  ```bash
  python llava_captioning.py
  ```
## 3. Output
- Outputs are saved as a JSONL file (1 line = 1 video).
- It reads the existing output and skips already-processed video_name entries to resume (resume).

## Notes
- llava_captioning.py assumes the following structure:
  - /home/dataset/video_eval/L{1..5}/{short,medium,long}/*.mp4
