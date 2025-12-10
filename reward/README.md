# A Clean Test Script

## 🛠️ Usage

## 1. Download the checkpoint
Download the pre-trained OmniAID checkpoint ([checkpoint_mirage.pth](https://huggingface.co/Yunncheng/OmniAID/blob/main/checkpoint_mirage.pth)) and update config.ckpt_path in clean_test.py.

## 2. Prepare your images
Modify the image_paths list in main() to point to your local image files (PNG/JPG).

## 3. Run the test
```python
python clean_test.py
```
The script will output a list of detection scores (higher = more likely AI-generated).

