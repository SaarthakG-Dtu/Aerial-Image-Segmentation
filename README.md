# Aerial Image Segmentation

FastAPI-backed aerial semantic segmentation app using transfer learning with an ImageNet-pretrained Inception-v4 encoder and U-Net decoder.

## Repo Layout

- `aerial_segmentation/` - Python package containing the training, dataset, model, metrics, and API implementation.
- `main.py` - thin training wrapper for `python main.py`.
- `app.py` - thin FastAPI wrapper for `python -m uvicorn app:app`.
- `frontend.html` - upload/demo UI.
- `classes_dataset/` - original images and semantic label masks.
- `outputs/` - generated checkpoints, plots, logs, and inference samples.
- `Notebooks/` - reference notebooks only; runtime code does not depend on them.

## Train

```bash
python main.py --epochs 15 --batch_size 1 --img_height 256 --img_width 384 --freeze_encoder_epochs 2
```

For higher-quality training on a GPU, increase `--img_height`, `--img_width`, and `--batch_size`.
On DigitalOcean GPU droplets, install dependencies first:

```bash
pip install -r requirements.txt
```

## Serve

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

The API loads `outputs/final_model.pt` first, then `outputs/best_model.pt`. If neither exists, it runs in untrained demo mode.
