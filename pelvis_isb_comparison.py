#!/usr/bin/env python3
"""Compare an ISB pelvis frame built from Motive markers with a BVH ``Hips`` segment.

Pelvis anatomical frame (Wu et al. 2002, J Biomech 35:543-548, section 4.1):

- Z: parallel to the line joining the ASISs, pointing to the right.
- X: parallel to the plane defined by both ASISs and the PSIS midpoint,
  orthogonal to Z, pointing anteriorly.
- Y: orthogonal to X and Z, pointing cranially (Y = Z x X).
- Origin: ISB places it at the hip joint centre (not at the PSIS midpoint).  Hip joint centres are not
  measured markers, so this script uses the ASIS midpoint as the reported origin.
  The origin has no effect on the orientation or on the Euler angles.

Pelvis orientation relative to the laboratory is decomposed with the intrinsic
ZXY sequence ``R = Rz(alpha) @ Rx(beta) @ Ry(gamma)`` after the laboratory axes
have been expressed with the ISB convention (X anterior, Y up, Z right).  With the
right-hand rule:

- alpha (about Z): positive = X anterior rotates upward = posterior tilt.
- beta (about X, floating axis): positive = right side moves down (Y tilts to the right).
- gamma (about Y): positive = right side moves anteriorly (pelvis turns to the left).

The BVH ``Hips`` local axes are not anatomical axes.  ``--bvh-hips-axes`` gives
the fixed mapping used to express them with ISB axis names (Motive default:
X = left, Y = up, Z = forward).  The comparison reports both the raw difference
and the constant offset rotation between the two frames, which is a definition
difference rather than a measurement error.

Sequence choice: Wu et al. (2002) prescribe ZXY for the hip JCS (pelvis -> femur);
they do not prescribe a sequence for the pelvis relative to the laboratory.  ZXY
relative to the lab is only interpretable when the subject faces the lab forward
axis: when the subject turns (e.g. 180 deg), tilt and obliquity change sign.  For
trials with large heading changes use ``--sequence YXZ`` (Baker 2001, Gait Posture
13:1-6), which puts the heading first.

Example (Motive exports: C3D lab is Z-up in mm, BVH world is Y-up in cm)::

    python pelvis_isb_comparison.py         --c3d data/2026-06-30_P6_flat/Motive/P6_Static.c3d         --bvh "data/2026-06-30_P6_flat/Motive/P6_Static_Skeleton 001.bvh"         --lab-axes "X=left,Y=backward,Z=up" --bvh-world-axes "X=right,Y=up,Z=backward"         --out-dir out_pelvis_isb/P6_Static

Use the lab axes resolved on the static trial (printed in ``conventions``) for the
dynamic trials of the same session, so all trials share one laboratory frame.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PELVIS_MARKERS = ("LIAS", "RIAS", "LIPS", "RIPS")
ISB_SEQUENCE = "ZXY"
AXIS_ANGLE_NAMES = {"Z": "tilt_Z", "X": "obliquity_X", "Y": "rotation_Y"}
AUTO_AXES_WINDOW_S = 1.0  # "auto" lab axes use the first second (subject in starting posture)
AXIS_WORDS = {
    "right": "+R", "left": "-R",
    "up": "+U", "down": "-U",
    "forward": "+F", "backward": "-F",
}


# --------------------------------------------------------------------------- axes


def parse_axis_spec(spec: str) -> np.ndarray:
    """Return the matrix whose columns are the ISB axes (X fwd, Y up, Z right) in source axes.

    ``spec`` names the anatomical direction of each source axis, e.g.
    ``"X=left,Y=up,Z=forward"``.
    """

    isb_index = {"F": 0, "U": 1, "R": 2}
    source_to_isb = np.zeros((3, 3))
    seen = set()
    for item in spec.split(","):
        axis, word = (part.strip() for part in item.split("="))
        axis = axis.upper()
        if axis not in "XYZ" or word.lower() not in AXIS_WORDS:
            raise ValueError(f"Invalid axis specification {item!r}")
        sign, direction = AXIS_WORDS[word.lower()]
        source_to_isb[isb_index[direction], "XYZ".index(axis)] = 1.0 if sign == "+" else -1.0
        seen.add(direction)
    if len(seen) != 3:
        raise ValueError(f"Axis specification {spec!r} must name forward, up and right once each")
    if not np.isclose(np.linalg.det(source_to_isb), 1.0):
        raise ValueError(f"Axis specification {spec!r} is left-handed")
    # rows = ISB axes expressed in source coordinates, so the transpose holds them as columns
    return source_to_isb.T


def lab_axes_from_forward(forward: str, up: str = "+Y") -> str:
    """Build a lab axis specification from the lab axes pointing forward and up."""

    def split(token: str) -> tuple[str, str]:
        token = token.strip().upper()
        return (token[0] if token[0] in "+-" else "+"), token[-1]

    f_sign, f_axis = split(forward)
    u_sign, u_axis = split(up)
    if f_axis == u_axis:
        raise ValueError("Forward and up must be different lab axes")
    f_vec = np.zeros(3); f_vec["XYZ".index(f_axis)] = 1 if f_sign == "+" else -1
    u_vec = np.zeros(3); u_vec["XYZ".index(u_axis)] = 1 if u_sign == "+" else -1
    r_vec = np.cross(f_vec, u_vec)  # forward x up = right (right-handed)
    words = {}
    for vec, pos, neg in ((f_vec, "forward", "backward"), (u_vec, "up", "down"), (r_vec, "right", "left")):
        axis = int(np.flatnonzero(vec)[0])
        words["XYZ"[axis]] = pos if vec[axis] > 0 else neg
    return ",".join(f"{axis}={words[axis]}" for axis in "XYZ")


def axes_spec_from_directions(forward: np.ndarray, up: np.ndarray) -> str:
    """Snap measured forward/up directions to the closest lab axes and return an axis specification.

    Only a permutation/sign change of the lab axes is chosen, so any residual
    heading or lean of the subject stays in the angles.
    """

    def token(vector: np.ndarray, allowed: str) -> str:
        axis = max(allowed, key=lambda a: abs(vector["XYZ".index(a)]))
        return ("+" if vector["XYZ".index(axis)] >= 0 else "-") + axis

    up_token = token(up, "XYZ")
    forward_token = token(forward, "XYZ".replace(up_token[1], ""))
    return lab_axes_from_forward(forward_token, up_token)


# ------------------------------------------------------------------------ markers


def load_c3d_markers(path: Path, names: tuple[str, ...]) -> tuple[dict[str, np.ndarray], float, str]:
    """Load markers as ``(n_frames, 3)`` arrays in mm, matching labels by suffix."""

    import ezc3d

    c3d = ezc3d.c3d(str(path))
    point = c3d["parameters"]["POINT"]
    labels = [label.strip() for label in point["LABELS"]["value"]]
    units = point.get("UNITS", {}).get("value", ["mm"])[0].strip().lower()
    scale = {"mm": 1.0, "cm": 10.0, "m": 1000.0}.get(units)
    if scale is None:
        raise ValueError(f"Unsupported C3D point unit {units!r} in {path}")
    rate = float(point["RATE"]["value"][0])
    points = c3d["data"]["points"][:3] * scale

    markers = {}
    for name in names:
        matches = [i for i, label in enumerate(labels) if label == name or label.endswith(f"_{name}") or label.endswith(f":{name}")]
        if len(matches) != 1:
            raise KeyError(f"Marker {name!r}: expected one label, found {[labels[i] for i in matches]} in {path}")
        xyz = points[:, matches[0], :].T.copy()
        xyz[np.all(xyz == 0.0, axis=1)] = np.nan  # C3D convention for missing samples
        markers[name] = xyz
    return markers, rate, units


def isb_pelvis_frame(lasis, rasis, lpsis, rpsis) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, origin)``; ``R[f]`` has the pelvis X, Y, Z axes as columns in lab coordinates."""

    mid_asis = (lasis + rasis) / 2.0
    mid_psis = (lpsis + rpsis) / 2.0
    z = rasis - lasis
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    in_plane = mid_asis - mid_psis
    x = in_plane - np.sum(in_plane * z, axis=1, keepdims=True) * z
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=2), mid_asis


# ---------------------------------------------------------------------------- BVH


@dataclass
class BvhRoot:
    name: str
    rotations: np.ndarray  # (n_frames, 3, 3) root local axes in BVH world
    positions: np.ndarray  # (n_frames, 3) root translation, BVH units
    frame_time: float


def load_bvh_root(path: Path) -> BvhRoot:
    """Read the root joint channels of a BVH file (rotation order taken from the file)."""

    text = path.read_text()
    header, motion = text.split("MOTION", 1)
    tokens = header.split()
    root_name = tokens[tokens.index("ROOT") + 1]
    channel_at = tokens.index("CHANNELS")
    n_channels = int(tokens[channel_at + 1])
    root_channels = tokens[channel_at + 2: channel_at + 2 + n_channels]

    lines = [line for line in motion.strip().splitlines() if line.strip()]
    n_frames = int(lines[0].split(":")[1])
    frame_time = float(lines[1].split(":")[1])
    values = np.array([[float(v) for v in line.split()[:n_channels]] for line in lines[2: 2 + n_frames]])

    positions = np.zeros((n_frames, 3))
    angle_columns, sequence = [], ""
    for column, channel in enumerate(root_channels):
        if channel.endswith("position"):
            positions[:, "XYZ".index(channel[0])] = values[:, column]
        elif channel.endswith("rotation"):
            angle_columns.append(column)
            sequence += channel[0]
    # BVH convention: rotations applied in the listed order about the moving axes (intrinsic)
    rotations = Rotation.from_euler(sequence.upper(), values[:, angle_columns], degrees=True).as_matrix()
    return BvhRoot(root_name, rotations, positions, frame_time)


# --------------------------------------------------------------------- comparison


def angle_names(sequence: str) -> tuple[str, ...]:
    return tuple(AXIS_ANGLE_NAMES[axis] for axis in sequence)


def euler_deg(rotations: np.ndarray, sequence: str = ISB_SEQUENCE) -> np.ndarray:
    """Intrinsic Euler angles (deg), unwrapped over time; NaN where the rotation is undefined."""

    out = np.full((len(rotations), 3), np.nan)
    valid = np.all(np.isfinite(rotations), axis=(1, 2))
    if valid.any():
        angles = Rotation.from_matrix(rotations[valid]).as_euler(sequence, degrees=True)
        out[valid] = np.degrees(np.unwrap(np.radians(angles), axis=0))
    return out


def wrap_deg(angles: np.ndarray) -> np.ndarray:
    return (angles + 180.0) % 360.0 - 180.0


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(rotations), axis=(1, 2))
    return Rotation.from_matrix(rotations[valid]).mean().as_matrix()


def compare(markers_c3d: Path, bvh_path: Path, lab_axes: str, bvh_world_axes: str | None,
            bvh_hips_axes: str, out_dir: Path, sequence: str = ISB_SEQUENCE) -> dict:
    sequence = sequence.upper()
    if sorted(sequence) != ["X", "Y", "Z"]:
        raise ValueError(f"Sequence {sequence!r} must be a Cardan sequence using X, Y and Z once")
    names = angle_names(sequence)
    markers, rate, units = load_c3d_markers(markers_c3d, PELVIS_MARKERS)
    r_lab_pelvis, origin = isb_pelvis_frame(*(markers[name] for name in PELVIS_MARKERS))

    hips_to_isb = parse_axis_spec(bvh_hips_axes)      # columns = anatomical axes in Hips coords
    bvh = load_bvh_root(bvh_path)
    bvh_anatomical_world = bvh.rotations @ hips_to_isb

    # "auto": each source's lab axes are snapped to the subject's mean pelvis axes
    if lab_axes == "auto":
        start = slice(0, max(1, int(AUTO_AXES_WINDOW_S * rate)))
        lab_axes = axes_spec_from_directions(np.nanmean(r_lab_pelvis[start, :, 0], axis=0),
                                             np.nanmean(r_lab_pelvis[start, :, 1], axis=0))
    if bvh_world_axes in (None, "auto"):
        start = slice(0, max(1, int(AUTO_AXES_WINDOW_S / bvh.frame_time)))
        bvh_world_axes = axes_spec_from_directions(bvh_anatomical_world[start, :, 0].mean(axis=0),
                                                   bvh_anatomical_world[start, :, 1].mean(axis=0))
    lab_to_isb = parse_axis_spec(lab_axes)            # columns = ISB lab axes in lab coords
    bvh_world_to_isb = parse_axis_spec(bvh_world_axes)

    # Express everything in the ISB laboratory basis (X fwd, Y up, Z right)
    r_markers = lab_to_isb.T @ r_lab_pelvis
    r_bvh = bvh_world_to_isb.T @ bvh_anatomical_world

    n = min(len(r_markers), len(r_bvh))
    bvh_rate = 1.0 / bvh.frame_time
    warnings = []
    if len(r_markers) != len(r_bvh):
        warnings.append(f"Frame counts differ (C3D {len(r_markers)}, BVH {len(r_bvh)}); first {n} frames compared, same start assumed.")
    if abs(bvh_rate - rate) > 0.01 * rate:
        warnings.append(f"Sampling rates differ (C3D {rate:.3f} Hz, BVH {bvh_rate:.3f} Hz).")
    r_markers, r_bvh, origin = r_markers[:n], r_bvh[:n], origin[:n]

    angles_markers = euler_deg(r_markers, sequence)
    angles_bvh = euler_deg(r_bvh, sequence)
    difference = wrap_deg(angles_bvh - angles_markers)
    relative = np.einsum("fji,fjk->fik", r_markers, r_bvh)  # R_markers^T R_bvh
    geodesic = np.full(n, np.nan)
    valid = np.all(np.isfinite(relative), axis=(1, 2))
    geodesic[valid] = np.degrees(Rotation.from_matrix(relative[valid]).magnitude())

    offset = mean_rotation(relative)
    corrected = euler_deg(r_bvh @ offset.T, sequence)       # BVH with the constant offset removed
    residual = wrap_deg(corrected - angles_markers)

    # Where is the BVH root relative to the ASIS midpoint, in the marker pelvis frame (mm)?
    # Assumes both files share the lab origin (true for Motive C3D + Motive BVH exports).
    bvh_scale_to_mm = 10.0 if np.nanmax(np.abs(bvh.positions)) < 500 else 1.0  # heuristic: Motive BVH is in cm
    root_mm = (bvh_world_to_isb.T @ bvh.positions[:n].T).T * bvh_scale_to_mm
    origin_isb = (lab_to_isb.T @ origin.T).T
    root_in_pelvis = np.einsum("fji,fj->fi", r_markers, root_mm - origin_isb)
    root_distance = np.linalg.norm(root_in_pelvis, axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pelvis_angles.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_s",
                         *(f"markers_{a}_deg" for a in names),
                         *(f"bvh_{a}_deg" for a in names),
                         *(f"diff_{a}_deg" for a in names),
                         *(f"diff_offset_removed_{a}_deg" for a in names),
                         "geodesic_deg", "bvh_root_to_asis_mid_mm"])
        for f in range(n):
            writer.writerow([f, f / rate, *angles_markers[f], *angles_bvh[f], *difference[f],
                             *residual[f], geodesic[f], root_distance[f]])

    def stats(values: np.ndarray) -> dict:
        return {name: {"mean": float(np.nanmean(values[:, k])), "sd": float(np.nanstd(values[:, k])),
                       "rms": float(np.sqrt(np.nanmean(values[:, k] ** 2)))} for k, name in enumerate(names)}

    summary = {
        "inputs": {"markers_c3d": str(markers_c3d), "bvh": str(bvh_path), "c3d_units": units,
                   "c3d_rate_hz": rate, "bvh_rate_hz": bvh_rate, "frames_compared": n},
        "conventions": {"pelvis_frame": "Wu et al. 2002 s4.1 (origin reported at ASIS midpoint)",
                        "euler_sequence": f"intrinsic {sequence}", "lab_axes": lab_axes,
                        "bvh_world_axes": bvh_world_axes, "bvh_hips_axes": bvh_hips_axes,
                        "signs": {"tilt_Z": "+ = posterior tilt", "obliquity_X": "+ = right side down",
                                  "rotation_Y": "+ = right side forward (turn to the left)"}},
        "markers_angles_deg": stats(angles_markers),
        "bvh_angles_deg": stats(angles_bvh),
        "difference_bvh_minus_markers_deg": stats(difference),
        "constant_offset_rotation": {f"{sequence.lower()}_deg": Rotation.from_matrix(offset).as_euler(sequence, degrees=True).tolist(),
                                     "magnitude_deg": float(np.degrees(Rotation.from_matrix(offset).magnitude()))},
        "difference_after_offset_removed_deg": stats(residual),
        "geodesic_deg": {"mean": float(np.nanmean(geodesic)), "max": float(np.nanmax(geodesic))},
        "bvh_root_to_asis_mid_mm": {"distance_mean": float(np.nanmean(root_distance)),
                                    "distance_sd": float(np.nanstd(root_distance)),
                                    "in_pelvis_frame_xyz_mean": np.nanmean(root_in_pelvis, axis=0).tolist(),
                                    "note": "X anterior, Y cranial, Z right; same lab origin assumed"},
        "warnings": warnings,
    }
    (out_dir / "pelvis_summary.json").write_text(json.dumps(summary, indent=2))
    plot(angles_markers, angles_bvh, corrected, rate, names, sequence, out_dir / "pelvis_angles.png")
    return summary


def plot(markers: np.ndarray, bvh: np.ndarray, corrected: np.ndarray, rate: float,
         names: tuple[str, ...], sequence: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.arange(len(markers)) / rate
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(time, markers[:, k], label="Motive markers (ISB)")
        ax.plot(time, bvh[:, k], label="BVH Hips", alpha=0.8)
        ax.plot(time, corrected[:, k], "--", label="BVH Hips, offset removed", alpha=0.8)
        ax.set_ylabel(f"{names[k]} (deg)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Pelvis orientation vs lab, intrinsic {sequence} (frame: Wu et al. 2002)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c3d", type=Path, required=True, help="Motive C3D with LIAS, RIAS, LIPS, RIPS markers.")
    parser.add_argument("--bvh", type=Path, required=True, help="BVH whose root segment is the pelvis (Hips).")
    parser.add_argument("--lab-axes", default="auto",
                        help='Anatomical meaning of the C3D lab axes, e.g. "X=left,Y=backward,Z=up". '
                             '"auto" (default) snaps up/forward to the lab axes closest to the mean pelvis Y/X axes.')
    parser.add_argument("--bvh-world-axes", default="auto",
                        help='Same for the BVH world. "auto" (default) snaps to the mean BVH Hips anatomical axes.')
    parser.add_argument("--bvh-hips-axes", default="X=left,Y=up,Z=forward",
                        help="Anatomical meaning of the BVH Hips local axes (Motive default: X=left,Y=up,Z=forward).")
    parser.add_argument("--sequence", default=ISB_SEQUENCE,
                        help="Intrinsic Cardan sequence (default ZXY). For trials with large heading changes "
                             "(turning gait), YXZ (Baker 2001) keeps tilt/obliquity independent of heading.")
    parser.add_argument("--out-dir", type=Path, default=Path("out_pelvis_isb"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compare(args.c3d, args.bvh, args.lab_axes, args.bvh_world_axes, args.bvh_hips_axes, args.out_dir, args.sequence)
    print(json.dumps({key: summary[key] for key in (
        "conventions", "markers_angles_deg", "bvh_angles_deg", "difference_bvh_minus_markers_deg",
        "constant_offset_rotation", "difference_after_offset_removed_deg", "bvh_root_to_asis_mid_mm", "warnings")}, indent=2))
    print(f"Outputs written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
