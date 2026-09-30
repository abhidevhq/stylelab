# StyleLab — Neural Art Studio

StyleLab is a local web app for neural style transfer. Choose a photograph and a style image, adjust style intensity, and let a frozen, pretrained VGG19 feature extractor guide optimization of a new image.

The app binds to `127.0.0.1`. Uploaded images are processed by the local Python server, not sent to a hosted AI service. The page imports Google Fonts, however, so the browser may contact Google to load the fonts while online.

## What happens when you create an image

1. The page previews the two selected images and posts them with the style-intensity value to FastAPI.
2. The server checks the actual file contents, limits each image to 10 MB and 25 megapixels, and saves temporary copies in `uploads/`.
3. The style-transfer module fits each image inside a 256 × 256 canvas without stretching its aspect ratio. VGG19 extracts content and style features.
4. Adam updates the generated image pixels for 500 steps. VGG19 is frozen. The UI polls the local server and displays real optimization progress.
5. The generated PNG is stored in `results/`; the content-image padding is cropped away so the output keeps the photograph’s proportions.
6. Temporary input copies are removed. Upload leftovers older than one day and result PNGs older than 30 days are cleaned at server startup and before a new render.

VGG19 is **not trained by this app**. Its course-provided pretrained weights are loaded from `pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5`.

## Project layout

```text
stylelab/
├── nst/
│   ├── __init__.py
│   ├── image_validation.py     # Validates image bytes and size limits
│   ├── settings.py             # Shared image size and optimizer settings
│   └── style_transfer.py       # VGG19, losses, and pixel optimization
├── pretrained-model/
│   └── vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
├── results/                    # Generated PNGs; old results expire after 30 days
├── static/
│   ├── app.js                  # Upload previews, progress polling, API request
│   ├── index.html              # Page structure
│   └── style.css               # Visual design and responsive layout
├── tests/
│   └── test_image_validation.py
├── uploads/                    # Temporary inputs, removed after a render
├── gpu_check.py                # Optional TensorFlow device diagnostic
├── requirements.txt            # Pinned app dependencies
├── requirements-dev.txt        # App dependencies plus pytest
├── server.py                   # Local FastAPI app and progress endpoint
└── README.md
```

## Requirements

- Windows 64-bit and Python 3.11.
- The VGG19 weights file at the exact path shown above. It is about 80 MB in this copy of the project.
- The weights are intentionally excluded from GitHub; see [`pretrained-model/README.md`](pretrained-model/README.md) for setup instructions.
- The direct app dependencies in `requirements.txt` are pinned. For an exact lock of every transitive dependency, generate a lock file after installing and verifying this environment.

This project is configured to **run neural style transfer on the CPU**. The model code hides TensorFlow GPU devices to avoid the CPU/GPU placement error seen with the AMD DirectML setup. It configures 6 TensorFlow intra-op threads and 1 inter-op thread by default, appropriate as a starting point for the Ryzen 5 7530U. If the laptop gets too hot, try 4 intra-op threads. More threads are not automatically faster.

## Run it on Windows with PowerShell

Open the `stylelab` folder in VS Code. Choose **Terminal → New Terminal** and confirm the terminal is PowerShell with the project folder as its current directory.

### 1. Create and activate an isolated environment

```powershell
py --list
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell says scripts are disabled, allow activation in this terminal window and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The prompt should begin with `(.venv)`. This environment belongs to this project and does not replace Python in other projects.

### 2. Install the pinned packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To install the test tool too:

```powershell
python -m pip install -r requirements-dev.txt
```

### 3. Start StyleLab

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

When Uvicorn says it is running, open [http://127.0.0.1:8000](http://127.0.0.1:8000). Stop the server with **Ctrl+C** in the terminal.

### 4. Use the app

1. Choose a content photograph.
2. Choose a style image.
3. Set **Style intensity** from 10% to 100% (default: 100%).
4. Select **Create my artwork** and watch the progress bar.
5. Select **Save image** to download the generated PNG.

The browser sends the percentage as a fraction: 100% becomes `1.0`. The API handles one render at a time to avoid running several CPU-heavy optimizations simultaneously. The current UI shows progress, but does not support cancelling a render.

## CPU performance settings

Thread limits are set before the VGG19 model loads. In PowerShell, set them before launching Uvicorn:

```powershell
$env:STYLELAB_TF_INTRA_THREADS = "4"
$env:STYLELAB_TF_INTER_THREADS = "1"
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Use `4` intra-op threads if you prefer lower CPU heat; the default is `6`. Restart the server after changing these values. The optimization step uses `@tf.function` on CPU to reduce Python overhead; the first step may take longer while TensorFlow traces the function.

`gpu_check.py` is an optional TensorFlow device diagnostic. It does not describe the app's execution policy: StyleLab intentionally hides GPU devices for this CPU configuration.

## Algorithm in brief

- **Content loss:** compares VGG19 activations at `block5_conv4`.
- **Style loss:** compares Gram matrices from five VGG19 convolution layers.
- **Total loss:** uses `alpha = 5.0` and `beta = 200.0 × strength`.
- **Optimization:** Adam updates the generated image tensor; it does not update VGG19's weights.
- **Targets:** content features and style features are calculated once before the optimization loop and reused.
- **Resolution:** the longest input edge is fitted inside the 256 × 256 model canvas. The content output is cropped back to the fitted content bounds, so its aspect ratio is preserved; output resolution is limited to this preview-sized canvas.

## Tests

With `(.venv)` active and development dependencies installed, run:

```powershell
python -m pytest
```

The tests cover valid image bytes, unsupported/corrupt images, and file-size/pixel limits. They do not run a full neural style-transfer render.

## Current scope and known limits

- This is a local portfolio application, not a hardened public service. Do not expose it on a public network without a deployment security review, request-body limits, authentication/rate controls as appropriate, and resource monitoring.
- Results are retained for up to 30 days and then cleaned on startup or before a new render. Save any result you want to keep.
- Processing is limited to 256 × 256 to keep CPU use manageable. This is not suitable for full-resolution print output.
- The model uses the supplied pretrained VGG19 weights; it does not train a new VGG19 model.
- Before distributing the project or its weight file, check and document the course and weight-file usage terms.
- The app has real step progress, but no cancellation, persistent job queue, or progress history after the server restarts.
- The test suite checks image validation, not every API or neural-network behavior. Run it on the project's Python environment before relying on a new dependency installation.
- The UI loads Google Fonts from the internet; the uploaded images themselves are handled locally.

## Interview explanation

> “I built a local neural style-transfer app using pretrained VGG19 as a frozen feature extractor. I calculate content loss from one VGG layer and style loss from Gram matrices across multiple layers. Adam optimizes the generated image pixels, while VGG19 stays fixed. I connected that algorithm to a FastAPI endpoint and a browser UI, with upload validation, progress reporting, CPU performance controls, and tests for image validation.”

Be clear that VGG19's weights were pretrained and supplied separately from your app. Explain which parts you implemented and the trade-offs between style strength, image detail, runtime, and CPU heat.
