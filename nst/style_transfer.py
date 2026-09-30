from pathlib import Path
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from nst.settings import DEFAULT_STEPS, IMG_SIZE, LEARNING_RATE


def _read_thread_setting(name: str, default: int) -> int:
    # Supplying a string fallback makes the environment lookup non-optional
    # to both Python and VS Code's type checker.
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


# For this six-core CPU, use one worker per physical core and keep TensorFlow's
# inter-op pool small. Set these environment variables before starting Uvicorn
# to tune performance or reduce heat on a different computer.
INTRA_OP_THREADS = _read_thread_setting("STYLELAB_TF_INTRA_THREADS", 6)
INTER_OP_THREADS = _read_thread_setting("STYLELAB_TF_INTER_THREADS", 1)
tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)


# Keep the Andrew Ng course seed.
tf.random.set_seed(272)

# Find the course VGG19 weights file relative to this Python file.
VGG_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "pretrained-model"
    / "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

if not VGG_WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        "VGG19 weights file was not found at:\n"
        f"{VGG_WEIGHTS_PATH}"
    )

# Load the fixed VGG19 feature extractor once when the server starts.
vgg = tf.keras.applications.VGG19(
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights=None,
)
vgg.load_weights(str(VGG_WEIGHTS_PATH))
vgg.trainable = False


# These are the VGG19 layers used to measure visual style.
STYLE_LAYERS = [
    ("block1_conv1", 0.2),
    ("block2_conv1", 0.2),
    ("block3_conv1", 0.2),
    ("block4_conv1", 0.2),
    ("block5_conv1", 0.2),
]

# This layer is used to measure image content.
CONTENT_LAYER = [("block5_conv4", 1.0)]


def build_feature_model(base_model, layer_names):
    """Build a model that returns activations from selected VGG19 layers."""
    outputs = [
        base_model.get_layer(layer_name).output
        for layer_name, _weight in layer_names
    ]
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)


# Output order: five style layers, followed by one content layer.
vgg_model_outputs = build_feature_model(
    vgg,
    STYLE_LAYERS + CONTENT_LAYER,
)


def compute_content_cost(content_outputs, generated_outputs):
    """Measure how much the generated image differs in content features."""
    content_activation = content_outputs[-1]
    generated_activation = generated_outputs[-1]

    _, height, width, channels = generated_activation.shape

    content_unrolled = tf.reshape(
        content_activation,
        shape=(1, height * width, channels),
    )
    generated_unrolled = tf.reshape(
        generated_activation,
        shape=(1, height * width, channels),
    )

    squared_difference = tf.square(
        content_unrolled - generated_unrolled
    )

    return tf.reduce_sum(squared_difference) / (
        4.0 * height * width * channels
    )


def gram_matrix(features):
    """Summarize correlations between feature channels."""
    return tf.matmul(features, features, transpose_b=True)


def compute_layer_style_cost(style_activation, generated_activation):
    """Compare the style Gram matrices for one VGG19 layer."""
    _, height, width, channels = generated_activation.shape

    # Change from [batch, height, width, channels]
    # to [channels, height * width].
    style_features = tf.reshape(
        tf.transpose(style_activation, perm=(0, 3, 1, 2)),
        shape=(channels, height * width),
    )
    generated_features = tf.reshape(
        tf.transpose(generated_activation, perm=(0, 3, 1, 2)),
        shape=(channels, height * width),
    )

    style_gram = gram_matrix(style_features)
    generated_gram = gram_matrix(generated_features)

    gram_difference = tf.square(style_gram - generated_gram)

    return tf.reduce_sum(gram_difference) / (
        4.0 * (height * width * channels) ** 2
    )


def compute_style_cost(
    style_outputs,
    generated_outputs,
    style_layers=STYLE_LAYERS,
):
    """Combine the style costs from the selected VGG19 layers."""
    total_style_cost = tf.constant(0.0, dtype=tf.float32)

    # The final output is the content layer, so style uses all earlier outputs.
    style_activations = style_outputs[:-1]
    generated_activations = generated_outputs[:-1]

    for index, (_layer_name, layer_weight) in enumerate(style_layers):
        layer_cost = compute_layer_style_cost(
            style_activations[index],
            generated_activations[index],
        )
        total_style_cost += layer_weight * layer_cost

    return total_style_cost


def total_cost(content_cost, style_cost, alpha=5.0, beta=200.0):
    """Combine content and style losses."""
    return alpha * content_cost + beta * style_cost


def clip_0_1(image_tensor):
    """Keep image pixel values in the displayable 0-to-1 range."""
    return tf.clip_by_value(image_tensor, 0.0, 1.0)


def load_uploaded_image(image_path, preserve_aspect=True):
    """Load an image on a square canvas; return its inner crop bounds too."""
    with Image.open(image_path) as opened_image:
        image = ImageOps.exif_transpose(opened_image).convert("RGB")

        if preserve_aspect:
            image = ImageOps.contain(
                image,
                (IMG_SIZE, IMG_SIZE),
                method=Image.Resampling.LANCZOS,
            )
            left = (IMG_SIZE - image.width) // 2
            top = (IMG_SIZE - image.height) // 2
            bounds = (left, top, left + image.width, top + image.height)

            # Fill the unused canvas with the image's average color instead of
            # stretching the photo. The content result is cropped back later.
            average_color = image.resize(
                (1, 1), Image.Resampling.BOX
            ).getpixel((0, 0))
            canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), average_color)
            canvas.paste(image, (left, top))
            image = canvas
        else:
            image = image.resize(
                (IMG_SIZE, IMG_SIZE),
                Image.Resampling.LANCZOS,
            )
            bounds = (0, 0, IMG_SIZE, IMG_SIZE)

        image_array = np.asarray(image, dtype=np.uint8)

    # Add a batch dimension and convert pixel values from 0–255 to 0–1.
    image_batch = np.expand_dims(image_array, axis=0)
    image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.uint8)

    return tf.image.convert_image_dtype(image_tensor, tf.float32), bounds


def tensor_to_image(image_tensor):
    """Convert a [1, H, W, 3] image tensor into a PIL image."""
    image_tensor = clip_0_1(image_tensor)

    # Remove the batch dimension if present.
    if image_tensor.shape.rank == 4:
        image_tensor = image_tensor[0]

    pixels = tf.cast(
        tf.round(image_tensor * 255.0),
        tf.uint8,
    ).numpy()

    return Image.fromarray(pixels, mode="RGB")


def stylize_images(
    content_path,
    style_path,
    output_path,
    strength=0.65,
    steps=DEFAULT_STEPS,
    progress_callback=None,
):
    """
    Create a stylized image and save it.

    This is the function called by the web server.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1")

    content_image, content_bounds = load_uploaded_image(
        content_path,
        preserve_aspect=True,
    )
    style_image, _style_bounds = load_uploaded_image(
        style_path,
        preserve_aspect=True,
    )

    # These target features do not change during optimization.
    # Calculate them once per request, before the loop.
    content_targets = vgg_model_outputs(content_image, training=False)
    style_targets = vgg_model_outputs(style_image, training=False)

    # Start close to the content photo, with a small amount of noise.
    # The earlier value of +/-0.25 was quite noisy for a short run.
    generated_image = tf.Variable(content_image, trainable=True)
    noise = tf.random.uniform(
        shape=tf.shape(generated_image),
        minval=-0.05,
        maxval=0.05,
        dtype=tf.float32,
    )
    generated_image.assign(
        clip_0_1(generated_image + noise)
    )

    # Use a new optimizer for this request, so separate user uploads
    # do not share optimizer state.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    # Keep the course's content weight.
    # The UI strength value adjusts the style weight.
    strength = max(0.1, min(float(strength), 1.0))
    alpha = 5.0
    beta = 200.0 * strength

    # Compile the repeated CPU optimization step once, then reuse the graph.
    @tf.function(reduce_retracing=True)
    def train_step(image_to_update):
        with tf.GradientTape() as tape:
            # Only the generated image's features change each step.
            generated_outputs = vgg_model_outputs(
                image_to_update,
                training=False,
            )

            style_loss = compute_style_cost(
                style_targets,
                generated_outputs,
                style_layers=STYLE_LAYERS,
            )
            content_loss = compute_content_cost(
                content_targets,
                generated_outputs,
            )
            loss = total_cost(
                content_loss,
                style_loss,
                alpha=alpha,
                beta=beta,
            )

        # Update image pixels; VGG19 itself remains frozen.
        gradient = tape.gradient(loss, image_to_update)
        optimizer.apply_gradients(
            [(gradient, image_to_update)]
        )
        image_to_update.assign(
            clip_0_1(image_to_update)
        )

        return loss

    # Run optimization.
    for step in range(steps):
        loss = train_step(generated_image)
        completed_steps = step + 1

        if progress_callback and (
            completed_steps == 1
            or completed_steps % 5 == 0
            or completed_steps == steps
        ):
            progress_callback(
                completed_steps,
                steps,
                float(loss.numpy()),
            )

        # Printing loss periodically helps monitor progress.
        if step % 100 == 0 or step == steps - 1:
            print(
                f"Step {step + 1}/{steps} "
                f"- loss: {float(loss.numpy()):.2f}"
            )

    # Save the final image.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_image = tensor_to_image(generated_image)
    result_image = result_image.crop(content_bounds)
    result_image.save(output_path)

    return str(output_path)
