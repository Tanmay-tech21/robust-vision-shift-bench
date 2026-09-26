import numpy as np
import pytest

from vision_shift_bench import (
    CorruptionSpec,
    apply_corruption,
    make_reference_image,
    peak_signal_to_noise_ratio,
    root_mean_square_error,
)


@pytest.fixture
def image() -> np.ndarray:
    return make_reference_image(size=32)


@pytest.mark.parametrize(
    "kind", ["brightness", "contrast", "gaussian_noise", "gaussian_blur"]
)
def test_corruptions_preserve_shape_range_and_input(image: np.ndarray, kind: str) -> None:
    original = image.copy()
    corrupted = apply_corruption(image, CorruptionSpec(kind=kind, severity=3, seed=7))

    assert corrupted.shape == image.shape
    assert np.all((0.0 <= corrupted) & (corrupted <= 1.0))
    np.testing.assert_array_equal(image, original)


def test_noise_is_deterministic_and_scales_one_field(image: np.ndarray) -> None:
    mild = apply_corruption(image, CorruptionSpec("gaussian_noise", 1, seed=17))
    repeated = apply_corruption(image, CorruptionSpec("gaussian_noise", 1, seed=17))
    severe = apply_corruption(image, CorruptionSpec("gaussian_noise", 5, seed=17))

    np.testing.assert_array_equal(mild, repeated)
    assert root_mean_square_error(image, severe) > root_mean_square_error(image, mild)


@pytest.mark.parametrize("kind", ["brightness", "contrast"])
def test_photometric_distortion_increases_with_severity(image: np.ndarray, kind: str) -> None:
    errors = [
        root_mean_square_error(
            image, apply_corruption(image, CorruptionSpec(kind=kind, severity=level))
        )
        for level in range(1, 6)
    ]
    assert errors == sorted(errors)
    assert len(set(errors)) == 5


def test_grayscale_and_single_channel_images_are_supported() -> None:
    grayscale = make_reference_image(size=24)[..., 0]
    single_channel = grayscale[..., None]

    assert apply_corruption(grayscale, CorruptionSpec("gaussian_blur", 2)).shape == grayscale.shape
    assert (
        apply_corruption(single_channel, CorruptionSpec("gaussian_blur", 2)).shape
        == single_channel.shape
    )


@pytest.mark.parametrize("severity", [0, 6, 1.5, True])
def test_invalid_severity_is_rejected(severity: object) -> None:
    with pytest.raises(ValueError, match="severity"):
        CorruptionSpec("brightness", severity)  # type: ignore[arg-type]


def test_invalid_pixels_are_rejected(image: np.ndarray) -> None:
    image[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_corruption(image, CorruptionSpec("brightness", 1))


def test_distortion_metrics_have_expected_limits(image: np.ndarray) -> None:
    assert root_mean_square_error(image, image) == 0.0
    assert peak_signal_to_noise_ratio(image, image) == float("inf")

    changed = apply_corruption(image, CorruptionSpec("brightness", 1))
    assert root_mean_square_error(image, changed) > 0.0
    assert np.isfinite(peak_signal_to_noise_ratio(image, changed))
