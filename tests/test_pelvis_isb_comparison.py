import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pelvis_isb_comparison import (
    axes_spec_from_directions,
    euler_deg,
    isb_pelvis_frame,
    lab_axes_from_forward,
    load_bvh_root,
    parse_axis_spec,
)

# Pelvis landmarks (mm) in an ISB-aligned lab: X anterior, Y up, Z right.
NEUTRAL = {
    "LIAS": np.array([0.0, 0.0, -120.0]),
    "RIAS": np.array([0.0, 0.0, 120.0]),
    "LIPS": np.array([-150.0, 0.0, -40.0]),
    "RIPS": np.array([-150.0, 0.0, 40.0]),
}


def rotated_landmarks(rotation: Rotation) -> list[np.ndarray]:
    return [rotation.apply(NEUTRAL[name])[None, :] for name in ("LIAS", "RIAS", "LIPS", "RIPS")]


class PelvisFrameTests(unittest.TestCase):
    def test_neutral_pelvis_is_identity(self):
        frame, origin = isb_pelvis_frame(*rotated_landmarks(Rotation.identity()))
        np.testing.assert_allclose(frame[0], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(origin[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_frame_is_right_handed_and_recovers_zxy_angles(self):
        expected = np.array([-12.0, 4.0, 20.0])
        rotation = Rotation.from_euler("ZXY", expected, degrees=True)
        frame, _ = isb_pelvis_frame(*rotated_landmarks(rotation))
        self.assertAlmostEqual(np.linalg.det(frame[0]), 1.0, places=12)
        np.testing.assert_allclose(euler_deg(frame)[0], expected, atol=1e-9)

    def test_anterior_tilt_is_negative_about_z(self):
        # ASISs move down relative to PSISs: anterior pelvic tilt
        landmarks = {k: v.copy() for k, v in NEUTRAL.items()}
        landmarks["LIAS"][1] -= 30.0
        landmarks["RIAS"][1] -= 30.0
        frame, _ = isb_pelvis_frame(*(landmarks[n][None, :] for n in ("LIAS", "RIAS", "LIPS", "RIPS")))
        self.assertLess(euler_deg(frame)[0, 0], 0.0)


class AxisSpecTests(unittest.TestCase):
    def test_identity_spec(self):
        np.testing.assert_allclose(parse_axis_spec("X=forward,Y=up,Z=right"), np.eye(3))

    def test_motive_c3d_lab_maps_to_isb(self):
        lab_to_isb = parse_axis_spec("X=left,Y=backward,Z=up")
        # ISB X (forward) is lab -Y, ISB Y (up) is lab +Z, ISB Z (right) is lab -X
        np.testing.assert_allclose(lab_to_isb[:, 0], [0, -1, 0])
        np.testing.assert_allclose(lab_to_isb[:, 1], [0, 0, 1])
        np.testing.assert_allclose(lab_to_isb[:, 2], [-1, 0, 0])

    def test_left_handed_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_axis_spec("X=right,Y=up,Z=forward")

    def test_auto_axes_snap_to_closest_lab_axes(self):
        spec = axes_spec_from_directions(forward=np.array([0.1, -0.98, 0.05]), up=np.array([0.0, 0.05, 1.0]))
        self.assertEqual(spec, "X=left,Y=backward,Z=up")
        self.assertEqual(lab_axes_from_forward("+X", "+Y"), "X=forward,Y=up,Z=right")


class BvhTests(unittest.TestCase):
    def test_root_rotation_uses_listed_channel_order(self):
        text = (
            "HIERARCHY\nROOT Hips\n{\n OFFSET 0 0 0\n"
            " CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
            " End Site\n {\n  OFFSET 0 1 0\n }\n}\n"
            "MOTION\nFrames: 1\nFrame Time: 0.008333\n1 2 3 10 20 30\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "root.bvh"
            path.write_text(text)
            root = load_bvh_root(path)
        expected = Rotation.from_euler("ZXY", [10, 20, 30], degrees=True).as_matrix()
        np.testing.assert_allclose(root.rotations[0], expected, atol=1e-12)
        np.testing.assert_allclose(root.positions[0], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
