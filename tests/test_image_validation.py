from io import BytesIO

import pytest
from PIL import Image

from nst.image_validation import ImageValidationError, validate_image_bytes


def make_image_bytes(image_format="PNG", size=(16, 12)):
    image = Image.new("RGB", size, color=(120, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_accepts_a_valid_png():
    image_bytes = make_image_bytes()

    assert validate_image_bytes(
        image_bytes,
        "Content",
        max_bytes=1024 * 1024,
        max_pixels=1000,
    )


def test_rejects_empty_upload():
    with pytest.raises(ImageValidationError, match="empty"):
        validate_image_bytes(
            b"",
            "Style",
            max_bytes=1024,
            max_pixels=1000,
        )


def test_rejects_upload_over_byte_limit():
    image_bytes = make_image_bytes()

    with pytest.raises(ImageValidationError) as error:
        validate_image_bytes(
            image_bytes,
            "Content",
            max_bytes=len(image_bytes) - 1,
            max_pixels=1000,
        )

    assert error.value.status_code == 413


def test_rejects_upload_over_pixel_limit():
    with pytest.raises(ImageValidationError) as error:
        validate_image_bytes(
            make_image_bytes(size=(16, 12)),
            "Style",
            max_bytes=1024 * 1024,
            max_pixels=100,
        )

    assert error.value.status_code == 413


def test_rejects_data_that_is_not_an_image():
    with pytest.raises(ImageValidationError, match="valid, supported image"):
        validate_image_bytes(
            b"not actually a PNG file",
            "Content",
            max_bytes=1024,
            max_pixels=1000,
        )


def test_rejects_unsupported_image_format():
    with pytest.raises(ImageValidationError, match="JPG, PNG, or WebP"):
        validate_image_bytes(
            make_image_bytes("BMP"),
            "Style",
            max_bytes=1024 * 1024,
            max_pixels=1000,
        )
