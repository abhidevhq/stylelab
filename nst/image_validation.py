"""Small, independently testable checks for uploaded image bytes."""

from io import BytesIO
import warnings

from PIL import Image, UnidentifiedImageError


ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def validate_image_bytes(image_bytes, label, max_bytes, max_pixels):
    """Validate size, real image format, pixel count, and decodability."""
    if not image_bytes:
        raise ImageValidationError(f"{label} image is empty.")

    if len(image_bytes) > max_bytes:
        raise ImageValidationError(
            f"{label} image must be {max_bytes // (1024 * 1024)} MB or smaller.",
            status_code=413,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in ALLOWED_IMAGE_FORMATS:
                    raise ImageValidationError(
                        f"{label} must be a JPG, PNG, or WebP image."
                    )

                width, height = image.size
                if width * height > max_pixels:
                    megapixels = max_pixels // 1_000_000
                    raise ImageValidationError(
                        f"{label} image must be {megapixels} megapixels or smaller.",
                        status_code=413,
                    )

                image.verify()
    except ImageValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        ValueError,
    ) as error:
        raise ImageValidationError(
            f"{label} is not a valid, supported image."
        ) from error

    return True
