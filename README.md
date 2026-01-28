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
