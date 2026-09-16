from pcr_cfar.cfar import ca_cfar_threshold, detect
from pcr_cfar.sar_scene import SceneConfig, render_scene


def test_scene_shapes():
    scene = render_scene(SceneConfig(height=64, width=80, n_targets=2, win=11, guard=3), seed=1)
    assert scene["img"].shape == (64, 80)
    assert scene["mask"].sum() >= 1


def test_ca_cfar_homogeneous_pfa_order():
    import numpy as np

    rng = np.random.default_rng(0)
    img = rng.exponential(1.0, size=(128, 128)).astype(np.float64)
    t = ca_cfar_threshold(img, win=21, guard=5, pfa=1e-3)
    pfa = float(detect(img, t)[10:-10, 10:-10].mean())
    assert 1e-5 < pfa < 2e-2
