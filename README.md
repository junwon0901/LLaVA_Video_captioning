# LLaVA-NeXT Video Inference (LLaVA-Video-72B-Qwen2)
* (lmms-lab/LLaVA-Video-72B-Qwen2)

---

## 1. 환경 설정


### 1. Clone this repository and navigate to the LLaVA folder

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
```
### 2. Install the inference package

```bash
conda create -n llava python=3.10 -y
conda activate llava
```

```bash
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

## 2. Captioning 실행
```bash
python video_caption.py
```