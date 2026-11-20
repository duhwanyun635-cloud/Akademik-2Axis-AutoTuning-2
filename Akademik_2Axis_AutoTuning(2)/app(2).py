"""Akademik Tryoshnikov 2-axis stabilization simulator + automatic tuning (2nd independent app).

Modes
- Virtual: synthetic wave -> simplified ship motion -> simulated IMU -> complementary filter
           -> FF+PD -> servo model -> stabilized plate.
- Actual ACE voyage: automatically downloads the complete public daily INS archive and the
  reconstructed sea-state CSV from Zenodo, indexes all daily files, then replays every continuous
  segment in chronological order without asking the user to upload files one by one.
- Auto tuning: searches FF, Kp, Kd and complementary-filter alpha using representative windows
  sampled across all available daily INS files. Servo parameters and assumed sensor-error settings
  are held fixed during tuning.

Important limitations
- ACE INS is uncorrected / not quality checked by the provider and was not synchronized with other
  onboard instrumentation as far as known.
- ACE INS is 1 Hz. Playback interpolates between samples to run the educational 50 Hz model;
  interpolation does not recover unrecorded high-frequency ship motion.
- ACE Hs/Tp is reconstructed from five-minute ship-heave spectra using a ship-specific RAO; it is
  not a direct buoy measurement at every instant.
- Auto-tuned parameters are model recommendations, not certified settings for a physical servo.
  Verify them on the built hardware with real 50 Hz IMU and actuator measurements before use.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Constants and paths
# -----------------------------------------------------------------------------
G = 9.81
CONTROL_DT = 0.02  # 50 Hz model integration
HISTORY_S = 30.0
SHIP = {
    "name": "R/V Akademik Tryoshnikov",
    "length": 133.53,
    "breadth": 23.25,
    "displacement": 16539,
    "draft": 8.5,
}
INS_URL = "https://zenodo.org/records/3733228/files/ace-inertial-navigation-csv.zip?download=1"
WAVE_URL = "https://zenodo.org/records/4541564/files/Reconstructed-Sea-State.csv?download=1"
INS_EXPECTED_SIZE = 646_925_085
INS_MD5 = "7778fe7e1a2472d0240022f47430e149"
WAVE_MD5 = "5fce25d202cf8588cc0963e2f0b2b835"
CACHE_DIR = Path.home() / ".cache" / "akademik_2axis"
INS_ARCHIVE = CACHE_DIR / "ace-inertial-navigation-csv.zip"
WAVE_CSV = CACHE_DIR / "Reconstructed-Sea-State.csv"
MANIFEST_JSON = CACHE_DIR / "ace_manifest.json"

SHIP_COLOR = "#F28C38"
PLATE_COLOR = "#2EC4B6"
PLATE_OFF_COLOR = "#E4572E"

SEA_STATES = {
    "잔잔한 파도": dict(Hs=0.5, Tp=4.0, direction=0, components=3, shock=3.0, gain=0.6),
    "거친 파도": dict(Hs=2.0, Tp=7.0, direction=25, components=5, shock=7.0, gain=2.0),
    "폭풍 파도": dict(Hs=4.0, Tp=10.0, direction=45, components=7, shock=12.0, gain=3.5),
}

ROLL_ALIASES = ["platform_roll", "roll", "roll_deg"]
PITCH_ALIASES = ["platform_pitch", "pitch", "pitch_deg"]
DATE_ALIASES = ["pc_date_utc", "date_utc", "date"]
TIME_ALIASES = ["pc_time_hhmmssss", "pc_time_hhmmsssss", "pc_time_hhmmss", "pc_time", "time_utc", "time"]


# -----------------------------------------------------------------------------
# Download and cache the complete official data once per app runtime/cache folder
# -----------------------------------------------------------------------------
def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_to_path(url: str, dest: Path, progress_cb: Callable[[int, int], None] | None = None,
                     expected_size: int | None = None, timeout: int = 240) -> None:
    """Download a large Zenodo file to a temporary file, then atomically rename it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Akademik-2Axis-Stabilizer/2.0", "Accept": "*/*"},
    )
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
            total = int(response.headers.get("Content-Length") or expected_size or 0)
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if progress_cb:
                    progress_cb(downloaded, total)
        if downloaded <= 0:
            raise IOError("서버에서 빈 파일을 받았습니다.")
        if expected_size and downloaded != expected_size:
            raise IOError(
                f"다운로드 크기가 예상과 다릅니다. 예상 {expected_size:,} bytes / 실제 {downloaded:,} bytes"
            )
        part.replace(dest)
    except Exception:
        try:
            part.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def dataset_ready() -> bool:
    try:
        return (
            INS_ARCHIVE.is_file() and INS_ARCHIVE.stat().st_size >= 600_000_000
            and WAVE_CSV.is_file() and WAVE_CSV.stat().st_size >= 500_000
        )
    except OSError:
        return False


def ensure_dataset(progress: st.delta_generator.DeltaGenerator | None = None) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    status = progress if progress is not None else st
    if not INS_ARCHIVE.exists() or INS_ARCHIVE.stat().st_size < 600_000_000:
        status.write("① 전체 INS 항해 기록 다운로드 중 (약 646.9MB)…")
        bar = status.progress(0.0)
        def cb(done, total):
            if total > 0:
                bar.progress(min(done / total, 1.0), text=f"INS 다운로드: {done/1e6:.0f} / {total/1e6:.0f} MB")
        download_to_path(INS_URL, INS_ARCHIVE, cb, INS_EXPECTED_SIZE)
        bar.progress(1.0, text="INS 다운로드 완료")
    if md5_file(INS_ARCHIVE) != INS_MD5:
        INS_ARCHIVE.unlink(missing_ok=True)
        raise IOError("INS ZIP의 MD5 검증에 실패했습니다. 파일을 삭제했으니 다시 다운로드해 주세요.")

    if not WAVE_CSV.exists() or WAVE_CSV.stat().st_size < 500_000:
        status.write("② 파고·주기 재구성 CSV 다운로드 중…")
        bar2 = status.progress(0.0)
        def cb2(done, total):
            if total > 0:
                bar2.progress(min(done / total, 1.0), text=f"파랑 데이터: {done/1e6:.1f} / {total/1e6:.1f} MB")
        download_to_path(WAVE_URL, WAVE_CSV, cb2)
        bar2.progress(1.0, text="파랑 데이터 다운로드 완료")
    if md5_file(WAVE_CSV) != WAVE_MD5:
        WAVE_CSV.unlink(missing_ok=True)
        raise IOError("파랑 CSV의 MD5 검증에 실패했습니다. 다시 다운로드해 주세요.")
    if not zipfile.is_zipfile(INS_ARCHIVE):
        raise IOError("다운로드한 INS 파일이 정상적인 ZIP 압축파일이 아닙니다.")
    return INS_ARCHIVE, WAVE_CSV


# -----------------------------------------------------------------------------
# Dataset parsing / chronological manifest
# -----------------------------------------------------------------------------
def find_col(columns, aliases):
    mapping = {str(c).replace("\ufeff", "").strip().lower(): c for c in columns}
    for a in aliases:
        if a.lower() in mapping:
            return mapping[a.lower()]
    return None


def read_ins_csv(file_obj) -> pd.DataFrame:
    try:
        raw = pd.read_csv(file_obj, low_memory=False)
    except Exception as exc:
        raise ValueError(f"INS CSV 읽기 실패: {exc}") from exc
    raw.columns = [str(c).replace("\ufeff", "").strip() for c in raw.columns]
    roll_col = find_col(raw.columns, ROLL_ALIASES)
    pitch_col = find_col(raw.columns, PITCH_ALIASES)
    date_col = find_col(raw.columns, DATE_ALIASES)
    time_col = find_col(raw.columns, TIME_ALIASES)
    stamp_col = find_col(raw.columns, ["timestamp", "datetime", "datetime_utc"])
    if roll_col is None or pitch_col is None:
        raise ValueError("platform_roll / platform_pitch 열을 찾지 못했습니다.")
    if stamp_col is not None:
        timestamp = pd.to_datetime(raw[stamp_col], utc=True, errors="coerce")
    elif date_col is not None and time_col is not None:
        timestamp = pd.to_datetime(raw[date_col].astype(str).str.strip() + " " +
                                    raw[time_col].astype(str).str.strip(), utc=True, errors="coerce")
    else:
        raise ValueError("UTC 시간 열(pc_date_utc, pc_time_hhmmssss)을 찾지 못했습니다.")

    def numeric(aliases):
        c = find_col(raw.columns, aliases)
        if c is None:
            return np.full(len(raw), np.nan, dtype=float)
        return pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=float)

    out = pd.DataFrame({
        "timestamp": timestamp,
        "ship_roll": pd.to_numeric(raw[roll_col], errors="coerce"),
        "ship_pitch": pd.to_numeric(raw[pitch_col], errors="coerce"),
        "heading": numeric(["heading"]),
        "north_speed": numeric(["north_speed_ms-1"]),
        "east_speed": numeric(["east_speed_ms-1"]),
        "vertical_speed": numeric(["vertical_speed_ms-1"]),
    })
    out = out.dropna(subset=["timestamp", "ship_roll", "ship_pitch"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if len(out) < 10:
        raise ValueError(f"유효한 자세 기록이 {len(out)}개뿐입니다.")
    if (out[["ship_roll", "ship_pitch"]].abs() > 90).any().any():
        raise ValueError("Roll/Pitch가 ±90°를 초과합니다. 열 이름과 단위를 확인해 주세요.")
    return out


def read_wave_csv(path_or_file) -> pd.DataFrame:
    raw = pd.read_csv(path_or_file, low_memory=False)
    raw.columns = [str(c).replace("\ufeff", "").strip() for c in raw.columns]
    tcol = find_col(raw.columns, ["Date_Time", "datetime", "timestamp", "time"])
    hcol = find_col(raw.columns, ["Hs", "significant_wave_height"])
    pcol = find_col(raw.columns, ["Tp", "peak_period"])
    if not all([tcol, hcol, pcol]):
        raise ValueError("파랑 CSV에 Date_Time, Hs, Tp 열이 없습니다.")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(raw[tcol], utc=True, errors="coerce"),
        "Hs": pd.to_numeric(raw[hcol], errors="coerce"),
        "Tp": pd.to_numeric(raw[pcol], errors="coerce"),
    }).dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out


def split_continuous(df: pd.DataFrame):
    ts = df["timestamp"].astype("int64").to_numpy() / 1e9
    dt = np.diff(ts)
    pos = dt[dt > 0]
    med = float(np.median(pos)) if len(pos) else 1.0
    gap_limit = max(2.5, med * 3.0)
    cuts = np.where(dt > gap_limit)[0] + 1
    bounds = np.r_[0, cuts, len(df)]
    result = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= 10:
            seg = df.iloc[a:b].copy().reset_index(drop=True)
            seg["elapsed_s"] = (seg["timestamp"] - seg["timestamp"].iloc[0]).dt.total_seconds()
            result.append(seg)
    return result, med, gap_limit


def list_ins_members(zf: zipfile.ZipFile) -> list[str]:
    pat = re.compile(r"ace_hydrins_\d{8}\.(?:csv|txt)$", re.IGNORECASE)
    return sorted([n for n in zf.namelist() if pat.search(Path(n).name)], key=lambda x: Path(x).name)


def build_manifest(archive_path: Path, progress_slot=None) -> list[dict]:
    cache_sig = {"archive_size": archive_path.stat().st_size, "archive_mtime": int(archive_path.stat().st_mtime)}
    if MANIFEST_JSON.exists():
        try:
            saved = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
            if saved.get("signature") == cache_sig and saved.get("segments"):
                return saved["segments"]
        except Exception:
            pass
    manifest = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = list_ins_members(zf)
        if not members:
            raise ValueError("ZIP 안에서 ace_hydrins_YYYYMMDD.csv/txt 파일을 찾지 못했습니다.")
        total = len(members)
        for i, member in enumerate(members, 1):
            with zf.open(member, "r") as f:
                daily = read_ins_csv(f)
            segments, med, gap = split_continuous(daily)
            for seg_i, seg in enumerate(segments):
                manifest.append({
                    "member": member,
                    "segment_index": int(seg_i),
                    "start": seg["timestamp"].iloc[0].isoformat(),
                    "end": seg["timestamp"].iloc[-1].isoformat(),
                    "rows": int(len(seg)),
                    "duration_s": float(seg["elapsed_s"].iloc[-1]),
                    "median_dt": float(med),
                })
            if progress_slot is not None:
                progress_slot.progress(i / total, text=f"항해 날짜 인덱싱: {i}/{total}일 파일")
    if not manifest:
        raise ValueError("연속된 항해 데이터 구간을 찾지 못했습니다.")
    manifest.sort(key=lambda x: (x["start"], x["member"], x["segment_index"]))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps({"signature": cache_sig, "segments": manifest}, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_manifest(archive_path: Path) -> list[dict]:
    return build_manifest(archive_path, None)


def load_segment(archive_path: Path, descriptor: dict) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path, "r") as zf:
        with zf.open(descriptor["member"], "r") as f:
            daily = read_ins_csv(f)
    segs, _, _ = split_continuous(daily)
    idx = int(descriptor["segment_index"])
    if idx >= len(segs):
        raise ValueError(f"데이터 구간을 다시 찾지 못했습니다: {descriptor['member']}")
    seg = segs[idx]
    x = seg["elapsed_s"].to_numpy(dtype=float)
    seg["roll_rate"] = np.gradient(seg["ship_roll"].to_numpy(dtype=float), x)
    seg["pitch_rate"] = np.gradient(seg["ship_pitch"].to_numpy(dtype=float), x)
    return seg


def nearest_wave(timestamp, wave_df: pd.DataFrame | None, tolerance_s=180.0):
    if wave_df is None or wave_df.empty or timestamp is None or pd.isna(timestamp):
        return None
    target = pd.Timestamp(timestamp)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    sec = wave_df["timestamp"].astype("int64").to_numpy() / 1e9
    delta = np.abs(sec - target.timestamp())
    i = int(np.argmin(delta))
    if float(delta[i]) > tolerance_s:
        return None
    row = wave_df.iloc[i]
    return {"Hs": float(row["Hs"]) if pd.notna(row["Hs"]) else np.nan,
            "Tp": float(row["Tp"]) if pd.notna(row["Tp"]) else np.nan,
            "timestamp": row["timestamp"], "delta_s": float(delta[i])}


# -----------------------------------------------------------------------------
# Motion and control models
# -----------------------------------------------------------------------------
def classify_sea_state(hs: float) -> tuple[int, str]:
    if not np.isfinite(hs): return 0, "자료 없음"
    if hs <= 0.1: return 1, "고요"
    if hs <= 0.5: return 2, "평온"
    if hs <= 1.25: return 3, "약간 거침"
    if hs <= 2.5: return 4, "보통"
    if hs <= 4.0: return 5, "거침"
    if hs <= 6.0: return 6, "매우 거침"
    return 7, "높음 이상"


def wave_components(tp: float, hs: float, count: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    fp = 1.0 / tp
    freqs = np.linspace(max(0.04, fp * 0.45), fp * 2.0, count)
    phases = rng.uniform(0, 2 * np.pi, count)
    spectrum = freqs**-5 * np.exp(-1.25 * (fp / freqs)**4)
    df = freqs[1] - freqs[0] if count > 1 else 0.1
    amps = np.sqrt(2 * spectrum * df)
    amps = amps / max(np.sqrt(np.sum(amps**2) / 2), 1e-9) * (hs / 4.0)
    spread = np.deg2rad(np.linspace(-18, 18, count))
    return freqs, amps, phases, spread


def ocean_surface(t: float, comps, direction: float, grid=(48, 34)):
    freqs, amps, phases, spread = comps
    X, Y = np.meshgrid(np.linspace(-75, 75, grid[0]), np.linspace(-18, 18, grid[1]))
    Z = np.zeros_like(X)
    th0 = np.deg2rad(direction)
    for f, a, ph, sp in zip(freqs, amps, phases, spread):
        w = 2 * np.pi * f
        k = w**2 / G
        th = th0 + sp
        Z += a * np.cos(k * (X*np.cos(th) + Y*np.sin(th)) - w*t + ph)
    return X, Y, Z


def shock_value(tau, peak, freq=0.5, zeta=0.25):
    if tau < 0: return 0.0, 0.0
    w = 2*np.pi*freq
    A = peak / 0.73
    e = np.exp(-zeta*w*tau)
    return A*e*np.sin(w*tau), A*e*(w*np.cos(w*tau)-zeta*w*np.sin(w*tau))


def virtual_motion(t, comps, direction, gain, shocks):
    freqs, amps, phases, spread = comps
    th0 = np.deg2rad(direction)
    roll = pitch = rr = pr = heave = ax = ay = 0.0
    for f, a, ph, sp in zip(freqs, amps, phases, spread):
        w = 2*np.pi*f; k = w*w/G; th = th0+sp; ang = -w*t+ph
        heave += a*np.cos(ang)
        slope = -a*k*np.sin(ang); slope_rate = a*k*w*np.cos(ang)
        roll += slope*np.sin(th); pitch += slope*np.cos(th)
        rr += slope_rate*np.sin(th); pr += slope_rate*np.cos(th)
        acc = a*w*w*np.sin(ang); ax += acc*np.cos(th); ay += acc*np.sin(th)
    scale = np.rad2deg(0.75)*gain
    roll*=scale; pitch*=scale; rr*=scale; pr*=scale
    for s in shocks:
        v, r = shock_value(t-s["t0"], s["peak"])
        if s["axis"] == "roll": roll += v; rr += r
        else: pitch += v; pr += r
    return {"roll": roll, "pitch": pitch, "roll_rate": rr, "pitch_rate": pr,
            "heave": heave, "ax": 0.5*ax, "ay": 0.5*ay,
            "heading": np.nan, "north_speed": np.nan, "east_speed": np.nan, "timestamp": None}


def actual_motion_at(t: float, df: pd.DataFrame):
    x = df["elapsed_s"].to_numpy(dtype=float)
    i = int(np.clip(np.searchsorted(x, t, side="left"), 0, len(x)-1))
    def interp(col):
        v = df[col].to_numpy(dtype=float)
        good = np.isfinite(v)
        if good.sum() < 2: return np.nan
        return float(np.interp(t, x[good], v[good]))
    return {"roll": interp("ship_roll"), "pitch": interp("ship_pitch"),
            "roll_rate": interp("roll_rate"), "pitch_rate": interp("pitch_rate"),
            "heave": 0.0, "ax": 0.0, "ay": 0.0,
            "heading": interp("heading"), "north_speed": interp("north_speed"),
            "east_speed": interp("east_speed"), "timestamp": df["timestamp"].iloc[i]}


def new_sim_state(initial=(0.0, 0.0), seed=7):
    rng = np.random.default_rng(seed)
    def axis(v):
        return {"base_est": float(v), "plate_est": float(v), "joint": 0.0, "joint_rate": 0.0,
                "base_bias": rng.normal(0, 1), "plate_bias": rng.normal(0, 1)}
    return {"t": 0.0, "roll": axis(initial[0]), "pitch": axis(initial[1]), "rng": rng,
            "history": [], "shocks": []}


def control_step(state, motion, p, controller_on=True, accel_disturb=0.0, dt=CONTROL_DT):
    row = {"time": state["t"], "heave": motion.get("heave", 0.0),
           "timestamp": motion.get("timestamp"), "heading": motion.get("heading", np.nan),
           "north_speed": motion.get("north_speed", np.nan), "east_speed": motion.get("east_speed", np.nan)}
    rng = state["rng"]
    acc_err = {"roll": np.rad2deg(np.arctan(motion.get("ay", 0.0)/G)),
               "pitch": -np.rad2deg(np.arctan(motion.get("ax", 0.0)/G))}
    if accel_disturb > 0:
        acc_err["roll"] += accel_disturb*np.sin(2*np.pi*0.23*state["t"])
        acc_err["pitch"] += accel_disturb*np.sin(2*np.pi*0.19*state["t"]+0.7)
    for axis in ("roll", "pitch"):
        s = state[axis]
        ship = float(motion[axis]); ship_rate = float(motion[f"{axis}_rate"])
        plate_before = ship + s["joint"]
        plate_rate = ship_rate + s["joint_rate"]
        gyro_b = ship_rate + p["gyro_bias"]*s["base_bias"] + rng.normal(0, p["gyro_noise"])
        acc_b = ship + acc_err[axis] + rng.normal(0, p["acc_noise"])
        s["base_est"] = p["alpha"]*(s["base_est"]+gyro_b*dt) + (1-p["alpha"])*acc_b
        gyro_p = plate_rate + p["gyro_bias"]*s["plate_bias"] + rng.normal(0, p["gyro_noise"])
        acc_p = plate_before + acc_err[axis] + rng.normal(0, p["acc_noise"])
        s["plate_est"] = p["alpha"]*(s["plate_est"]+gyro_p*dt) + (1-p["alpha"])*acc_p
        if controller_on:
            cmd = -p["ff"]*s["base_est"] - p["kp"]*s["plate_est"] - p["kd"]*gyro_p
        else: cmd = 0.0
        cmd = float(np.clip(cmd, -p["limit"], p["limit"]))
        d = (cmd-s["joint"])*dt/p["tau"]
        d = float(np.clip(d, -p["rate"]*dt, p["rate"]*dt))
        new_joint = float(np.clip(s["joint"]+d, -p["limit"], p["limit"]))
        s["joint_rate"] = (new_joint-s["joint"])/dt
        s["joint"] = new_joint
        row[f"ship_{axis}"] = ship
        row[f"plate_{axis}"] = ship + new_joint
        row[f"est_{axis}"] = s["base_est"]
        row[f"acc_{axis}"] = acc_b
        row[f"joint_{axis}"] = new_joint
    state["t"] += dt
    return row


# -----------------------------------------------------------------------------
# Auto tuning: 1 Hz sampled INS windows from every day, servo hardware fixed
# -----------------------------------------------------------------------------
TUNE_BOUNDS = {"ff": (0.0, 1.2), "kp": (0.0, 4.0), "kd": (0.0, 0.30), "alpha": (0.90, 0.999)}


def make_tuning_windows(archive_path: Path, manifest: list[dict], max_per_day=2, window_rows=240,
                        max_total_windows=180):
    """Sample representative 4-minute windows across dates; read each daily file once."""
    windows = []
    unique_members = sorted(set(d["member"] for d in manifest), key=lambda name: Path(name).name)
    member_rank = {member: i for i, member in enumerate(unique_members)}

    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in unique_members:
            with zf.open(member, "r") as f:
                daily = read_ins_csv(f)
            segments, _, _ = split_continuous(daily)
            candidates = []
            for seg in segments:
                n = len(seg)
                if n < 60:
                    continue
                length = min(window_rows, n)
                # Prefer start / middle / end coverage, but cap at max_per_day windows.
                possible = sorted(set(int(x) for x in np.linspace(0, max(0, n - length), min(max_per_day, 3))))
                for start_idx in possible:
                    window = seg.iloc[start_idx:start_idx + length].copy().reset_index(drop=True)
                    if len(window) < 60:
                        continue
                    elapsed = window["elapsed_s"].to_numpy(dtype=float)
                    window["roll_rate"] = np.gradient(window["ship_roll"].to_numpy(dtype=float), elapsed)
                    window["pitch_rate"] = np.gradient(window["ship_pitch"].to_numpy(dtype=float), elapsed)
                    candidates.append(window)
            # Limit daily contribution so long files do not dominate the objective.
            if len(candidates) > max_per_day:
                pick = np.linspace(0, len(candidates) - 1, max_per_day).round().astype(int)
                candidates = [candidates[i] for i in sorted(set(pick))]
            is_validation = (member_rank[member] % 5 == 0)
            for window in candidates:
                windows.append({"member": member, "validation": is_validation, "df": window})
            if len(windows) >= max_total_windows:
                break

    if not windows:
        raise ValueError("자동 최적화를 위한 1분 이상 연속 구간을 만들지 못했습니다.")

    train = [w["df"] for w in windows if not w["validation"]]
    valid = [w["df"] for w in windows if w["validation"]]
    # Small archives: split at window level, keeping at least one validation window.
    if len(train) < 2 or len(valid) < 1:
        train = [w["df"] for i, w in enumerate(windows) if i % 5 != 0]
        valid = [w["df"] for i, w in enumerate(windows) if i % 5 == 0]
    if not valid:
        valid = train[-max(1, len(train) // 5):]
    if not train:
        train = [w["df"] for w in windows]
    processed_days = len(set(w["member"] for w in windows))
    return train, valid, processed_days, len(windows)


def simulate_tuning_window(df: pd.DataFrame, p: dict, fixed: dict, seed=2026):
    """Run model on 1 Hz source points. No synthetic 50 Hz information is invented for tuning."""
    roll = df["ship_roll"].to_numpy(dtype=float)
    pitch = df["ship_pitch"].to_numpy(dtype=float)
    rr = df["roll_rate"].to_numpy(dtype=float)
    pr = df["pitch_rate"].to_numpy(dtype=float)
    t = df["elapsed_s"].to_numpy(dtype=float)
    n = len(t)
    if n < 5: return None
    dts = np.diff(t, prepend=t[0])
    positive = dts[dts > 0]
    default_dt = float(np.median(positive)) if len(positive) else 1.0
    dts[0] = default_dt
    dts = np.clip(dts, 0.2, 3.0)
    rng = np.random.default_rng(seed)
    # Constant sensor bias per virtual IMU; random measurement noise at each recorded sample.
    b_roll, bp_roll = rng.normal(0, 1, 2)
    b_pitch, bp_pitch = rng.normal(0, 1, 2)
    gb_r = rng.normal(0, fixed["gyro_noise"], n)
    gb_p = rng.normal(0, fixed["gyro_noise"], n)
    gp_r = rng.normal(0, fixed["gyro_noise"], n)
    gp_p = rng.normal(0, fixed["gyro_noise"], n)
    ab_r = rng.normal(0, fixed["acc_noise"], n)
    ab_p = rng.normal(0, fixed["acc_noise"], n)
    ap_r = rng.normal(0, fixed["acc_noise"], n)
    ap_p = rng.normal(0, fixed["acc_noise"], n)

    base_est_r, base_est_p = roll[0], pitch[0]
    plate_est_r, plate_est_p = roll[0], pitch[0]
    joint_r = joint_p = rate_r = rate_p = 0.0
    out_r = np.empty(n); out_p = np.empty(n)
    out_jr = np.empty(n); out_jp = np.empty(n)
    out_rdot = np.empty(n); out_pdot = np.empty(n)
    sat_count = 0
    for i in range(n):
        dt = float(dts[i])
        # alpha is specified per 20 ms controller tick; convert its equivalent weight to source dt.
        alpha_dt = float(p["alpha"] ** (dt / CONTROL_DT))
        ship_r, ship_p = roll[i], pitch[i]
        gyro_base_r = rr[i] + fixed["gyro_bias"]*b_roll + gb_r[i]
        gyro_base_p = pr[i] + fixed["gyro_bias"]*b_pitch + gb_p[i]
        t_rel = float(t[i] - t[0])
        disturb_r = fixed.get("accel_disturb", 0.0) * math.sin(2*math.pi*0.23*t_rel)
        disturb_p = fixed.get("accel_disturb", 0.0) * math.sin(2*math.pi*0.19*t_rel+0.7)
        acc_base_r = ship_r + disturb_r + ab_r[i]
        acc_base_p = ship_p + disturb_p + ab_p[i]
        base_est_r = alpha_dt*(base_est_r + gyro_base_r*dt) + (1-alpha_dt)*acc_base_r
        base_est_p = alpha_dt*(base_est_p + gyro_base_p*dt) + (1-alpha_dt)*acc_base_p

        plate_before_r = ship_r + joint_r
        plate_before_p = ship_p + joint_p
        gyro_plate_r = rr[i] + rate_r + fixed["gyro_bias"]*bp_roll + gp_r[i]
        gyro_plate_p = pr[i] + rate_p + fixed["gyro_bias"]*bp_pitch + gp_p[i]
        plate_est_r = alpha_dt*(plate_est_r + gyro_plate_r*dt) + (1-alpha_dt)*(plate_before_r+disturb_r+ap_r[i])
        plate_est_p = alpha_dt*(plate_est_p + gyro_plate_p*dt) + (1-alpha_dt)*(plate_before_p+disturb_p+ap_p[i])

        cmd_r = -p["ff"]*base_est_r - p["kp"]*plate_est_r - p["kd"]*gyro_plate_r
        cmd_p = -p["ff"]*base_est_p - p["kp"]*plate_est_p - p["kd"]*gyro_plate_p
        cmd_r = float(np.clip(cmd_r, -fixed["limit"], fixed["limit"]))
        cmd_p = float(np.clip(cmd_p, -fixed["limit"], fixed["limit"]))
        # Exact first-order servo response over the source sample interval + rate/travel limits.
        response = 1.0 - math.exp(-dt / max(fixed["tau"], 1e-4))
        move_r = float(np.clip((cmd_r-joint_r)*response, -fixed["rate"]*dt, fixed["rate"]*dt))
        move_p = float(np.clip((cmd_p-joint_p)*response, -fixed["rate"]*dt, fixed["rate"]*dt))
        new_jr = float(np.clip(joint_r+move_r, -fixed["limit"], fixed["limit"]))
        new_jp = float(np.clip(joint_p+move_p, -fixed["limit"], fixed["limit"]))
        rate_r = (new_jr-joint_r)/dt
        rate_p = (new_jp-joint_p)/dt
        joint_r, joint_p = new_jr, new_jp
        out_r[i] = ship_r + joint_r; out_p[i] = ship_p + joint_p
        out_jr[i] = joint_r; out_jp[i] = joint_p
        out_rdot[i] = rate_r; out_pdot[i] = rate_p
        if abs(joint_r) >= fixed["limit"]*0.999 or abs(joint_p) >= fixed["limit"]*0.999:
            sat_count += 1
    rms_ship_r = max(float(np.sqrt(np.mean(roll**2))), 0.1)
    rms_ship_p = max(float(np.sqrt(np.mean(pitch**2))), 0.1)
    rms_out_r = float(np.sqrt(np.mean(out_r**2)))
    rms_out_p = float(np.sqrt(np.mean(out_p**2)))
    p95_ship_r = max(float(np.percentile(np.abs(roll), 95)), 0.1)
    p95_ship_p = max(float(np.percentile(np.abs(pitch), 95)), 0.1)
    p95_out_r = float(np.percentile(np.abs(out_r), 95))
    p95_out_p = float(np.percentile(np.abs(out_p), 95))
    max_ship_r = max(float(np.max(np.abs(roll))), 0.1)
    max_ship_p = max(float(np.max(np.abs(pitch))), 0.1)
    max_out_r = float(np.max(np.abs(out_r)))
    max_out_p = float(np.max(np.abs(out_p)))
    effort = float((np.sqrt(np.mean(out_jr**2)) + np.sqrt(np.mean(out_jp**2))) / (2*fixed["limit"]))
    rate_usage = float((np.mean(np.abs(out_rdot)) + np.mean(np.abs(out_pdot))) / (2*fixed["rate"]))
    saturation = sat_count / n
    rms_ratio = 0.5*(rms_out_r/rms_ship_r + rms_out_p/rms_ship_p)
    p95_ratio = 0.5*(p95_out_r/p95_ship_r + p95_out_p/p95_ship_p)
    max_ratio = 0.5*(max_out_r/max_ship_r + max_out_p/max_ship_p)
    # Lower is better: RMS, 95th-percentile and peak attitude error dominate; actuator use is penalized.
    score = 0.45*rms_ratio + 0.20*p95_ratio + 0.15*max_ratio + 0.10*effort + 0.05*rate_usage + 0.05*saturation
    return {"score": score, "roll_rms": rms_out_r, "pitch_rms": rms_out_p,
            "roll_rms_before": rms_ship_r, "pitch_rms_before": rms_ship_p,
            "roll_p95": p95_out_r, "pitch_p95": p95_out_p,
            "roll_max": max_out_r, "pitch_max": max_out_p,
            "effort": effort, "rate_usage": rate_usage, "saturation": saturation,
            "roll_ratio": rms_out_r/rms_ship_r, "pitch_ratio": rms_out_p/rms_ship_p}


def evaluate_candidate(windows, p, fixed):
    results = [simulate_tuning_window(w, p, fixed, seed=2026+i) for i, w in enumerate(windows)]
    results = [r for r in results if r is not None]
    if not results: return {"score": float("inf")}
    # Average window scores so each window contributes equally, rather than longest files dominating.
    keys = ["score", "roll_rms", "pitch_rms", "roll_rms_before", "pitch_rms_before",
            "roll_p95", "pitch_p95", "roll_max", "pitch_max", "effort", "rate_usage", "saturation", "roll_ratio", "pitch_ratio"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


def optimize_params(train_windows, valid_windows, current, fixed, n_trials=36, progress_cb=None):
    rng = np.random.default_rng(42)
    candidates = [dict(current)]
    n_random = max(15, n_trials-9)
    for _ in range(n_random):
        candidates.append({k: float(rng.uniform(*TUNE_BOUNDS[k])) for k in TUNE_BOUNDS})
    scored = []
    total = len(candidates) + 8
    for i, candidate in enumerate(candidates):
        metrics = evaluate_candidate(train_windows, candidate, fixed)
        scored.append((metrics["score"], candidate, metrics))
        if progress_cb: progress_cb((i+1)/total, f"전역 탐색 {i+1}/{total}")
    scored.sort(key=lambda x: x[0])
    # Local refinement around the top four candidates.
    top = scored[:4]
    local_candidates = []
    for _, base, _ in top:
        for _ in range(2):
            c = {}
            for k, (lo, hi) in TUNE_BOUNDS.items():
                span = hi-lo
                c[k] = float(np.clip(base[k] + rng.normal(0, span*0.08), lo, hi))
            local_candidates.append(c)
    for j, candidate in enumerate(local_candidates):
        metrics = evaluate_candidate(train_windows, candidate, fixed)
        scored.append((metrics["score"], candidate, metrics))
        if progress_cb: progress_cb((len(candidates)+j+1)/total, f"상위 후보 정밀 탐색 {j+1}/{len(local_candidates)}")
    scored.sort(key=lambda x: x[0])
    best_score, best_params, train_metrics = scored[0]
    current_train = evaluate_candidate(train_windows, current, fixed)
    best_valid = evaluate_candidate(valid_windows, best_params, fixed)
    current_valid = evaluate_candidate(valid_windows, current, fixed)
    top_rows = []
    for rank, (score, pars, met) in enumerate(scored[:10], 1):
        top_rows.append({"순위": rank, "FF": pars["ff"], "Kp": pars["kp"], "Kd": pars["kd"],
                         "α": pars["alpha"], "학습 점수": score,
                         "Roll RMS": met["roll_rms"], "Pitch RMS": met["pitch_rms"],
                         "Roll 최대각": met["roll_max"], "Pitch 최대각": met["pitch_max"],
                         "서보 속도 사용률(%)": 100*met["rate_usage"],
                         "서보 포화율(%)": 100*met["saturation"]})
    return {"best": best_params, "train": train_metrics, "valid": best_valid,
            "current_train": current_train, "current_valid": current_valid,
            "top": pd.DataFrame(top_rows), "n_candidates": len(scored)}


def apply_best_parameters():
    result = st.session_state.get("optimization_result")
    if result:
        for key, value in result["best"].items():
            st.session_state[f"param_{key}"] = float(value)
        st.session_state["parameters_applied"] = True


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
BOX_FACES = [(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
             (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]


def box(cx, cy, cz, lx, ly, lz):
    V = np.array([[x,y,z] for x in (cx-lx/2,cx+lx/2) for y in (cy-ly/2,cy+ly/2) for z in (cz-lz/2,cz+lz/2)], float)
    f = np.array(BOX_FACES)
    return V, f[:,0], f[:,1], f[:,2]


def rotation(roll, pitch):
    r,p = np.deg2rad(roll), np.deg2rad(pitch)
    rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    return ry @ rx


def mesh(V, I, J, K, color, name, legend=True, opacity=1.0):
    return go.Mesh3d(x=V[:,0], y=V[:,1], z=V[:,2], i=I, j=J, k=K, color=color,
                     opacity=opacity, flatshading=True, name=name, showlegend=legend, hoverinfo="skip")


def hull_mesh():
    length, breadth = SHIP["length"], SHIP["breadth"]
    xs = np.linspace(-length/2, length/2, 45)
    vertices=[]
    for x in xs:
        ratio=abs(x/(length/2)); half=(breadth/2)*np.sqrt(max(0.015,1-ratio**2.3))
        vertices += [[x,-half,1.2],[x,-half*0.88,-0.35],[x,0,-1.8],[x,half*0.88,-0.35],[x,half,1.2]]
    ii=[];jj=[];kk=[]
    for i in range(len(xs)-1):
        for j in range(4):
            a=i*5+j;c=(i+1)*5+j
            ii += [a,a+1]; jj += [a+1,c+1]; kk += [c,c]
    return np.array(vertices),np.array(ii),np.array(jj),np.array(kk)

HULL = hull_mesh()


def scene_figure(row, actual_mode, comps=None, direction=0.0, controller_on=True):
    fig=go.Figure()
    if actual_mode:
        X,Y=np.meshgrid(np.linspace(-82,82,20),np.linspace(-24,24,12)); Z=np.full_like(X,-2.5)
        surf_name="수면(개념 표시 · 실측 파면 아님)"
    else:
        X,Y,Z=ocean_surface(float(row["time"]),comps,direction); surf_name="가상 파도"
    fig.add_trace(go.Surface(x=X,y=Y,z=Z,showscale=False,opacity=0.86,name=surf_name,hoverinfo="skip",
                             colorscale=[[0,"#042f55"],[0.5,"#087eae"],[1,"#a8e8f0"]]))
    V,I,J,K=HULL
    Rs=rotation(float(row["ship_roll"]),float(row["ship_pitch"]))
    lift=np.array([0,0,float(row.get("heave",0.0))])
    tf=lambda vertices:(Rs@vertices.T).T+lift
    fig.add_trace(mesh(tf(V),I,J,K,"#123f70","Akademik Tryoshnikov 선체"))
    structures=[box(39,0,3.1,20,12,3.6),box(42,0,6.1,10,9,2.5),box(31,0,5.5,2,2,6),box(31,0,8.8,5,0.6,0.4)]
    for idx,(v,i,j,k) in enumerate(structures): fig.add_trace(mesh(tf(v),i,j,k,"#f1f1e9","상부 구조물",legend=idx==0))
    pivot_ship=np.array([-42.0,0.0,1.7]); pivot=tf(pivot_ship[None,:])[0]
    post=box(0,0,1,1.3,1.3,2); v,i,j,k=post
    fig.add_trace(mesh((Rs@v.T).T+pivot,i,j,k,"#3b4a5a","짐벌 지지대"))
    Rp=rotation(float(row["plate_roll"]),float(row["plate_pitch"]))
    plate=box(0,0,2.15,2.8,1.7,0.18); v,i,j,k=plate
    fig.add_trace(mesh((Rp@v.T).T+pivot,i,j,k,PLATE_COLOR if controller_on else PLATE_OFF_COLOR,"안정화 상판"))
    cargo=box(0,0,3.1,0.55,0.55,1.7); v,i,j,k=cargo
    fig.add_trace(mesh((Rp@v.T).T+pivot,i,j,k,"#F2C14E","상판 적재물"))
    fig.update_layout(height=520,margin=dict(l=0,r=0,t=0,b=0),uirevision="ship-scene",
        legend=dict(orientation="h",yanchor="bottom",y=0.01,x=0.01,bgcolor="rgba(255,255,255,0.55)"),
        scene=dict(xaxis_title="선박 길이 방향(m)",yaxis_title="선박 폭 방향(m)",zaxis_title="높이(m)",
                   aspectmode="manual",aspectratio=dict(x=2.4,y=0.55,z=0.42),
                   camera=dict(eye=dict(x=-1.65,y=1.75,z=0.9)),bgcolor="#eaf5fb"))
    return fig


def angle_figure(df, axis, title):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["time"],y=df[f"ship_{axis}"],name="보정 전(선박 자세)",line=dict(color=SHIP_COLOR,width=1.6)))
    fig.add_trace(go.Scatter(x=df["time"],y=df[f"plate_{axis}"],name="보정 후(상판 자세)",line=dict(color=PLATE_COLOR,width=2.2)))
    fig.add_hrect(y0=-1,y1=1,fillcolor=PLATE_COLOR,opacity=0.08,line_width=0)
    fig.update_layout(title=title,height=260,margin=dict(l=10,r=10,t=40,b=10),yaxis_title="각도(°)",
                      xaxis_title="현재 구간 경과 시간(s)",hovermode="x unified",
                      legend=dict(orientation="h",y=1.15,x=1,xanchor="right"))
    return fig


def sensor_figure(df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["time"],y=df["acc_roll"],name="가속도계 환산각(하부)",line=dict(color="#9AA5B1",width=1)))
    fig.add_trace(go.Scatter(x=df["time"],y=df["est_roll"],name="상보필터 추정각",line=dict(color=PLATE_COLOR,width=2)))
    fig.add_trace(go.Scatter(x=df["time"],y=df["ship_roll"],name="INS Roll 기준",line=dict(color=SHIP_COLOR,width=1.4,dash="dot")))
    fig.update_layout(title="하부 IMU 모델: 측정값과 INS 자세 기준 비교(Roll)",height=260,
                      margin=dict(l=10,r=10,t=40,b=10),yaxis_title="각도(°)",xaxis_title="현재 구간 경과 시간(s)",hovermode="x unified")
    return fig


def live_performance(df, limit):
    rows=[]
    for axis,label in (("roll","Roll"),("pitch","Pitch")):
        s=df[f"ship_{axis}"].to_numpy(float); p=df[f"plate_{axis}"].to_numpy(float)
        sr=float(np.sqrt(np.mean(s*s))); pr=float(np.sqrt(np.mean(p*p)))
        rows.append({"축":label,"보정 전 RMS":f"{sr:.2f}°","보정 후 RMS":f"{pr:.2f}°",
                     "RMS 감소율":f"{100*(1-pr/sr):.1f}%" if sr>1e-8 else "-",
                     "보정 전 최대":f"{np.max(np.abs(s)):.2f}°","보정 후 최대":f"{np.max(np.abs(p)):.2f}°",
                     "서보 최대 사용각":f"{np.max(np.abs(df[f'joint_{axis}'])):.2f}° / ±{limit:.0f}°"})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Streamlit application
# -----------------------------------------------------------------------------
st.set_page_config(page_title="선박 2축 안정화 시뮬레이터 (2)",page_icon="🌊",layout="wide")
st.markdown("""
<style>
[data-testid="stMetric"] { padding: 6px 10px; border-left: 3px solid rgba(128,128,128,.30); }
[data-testid="stMetricLabel"] p { font-size: .85rem; }
</style>
""",unsafe_allow_html=True)
st.title("선박 2축 안정화 시뮬레이터 (2)")
st.caption("가상 파도 시험 + Akademik Tryoshnikov 전체 항해 기록 자동 재생 + 제어 파라미터 자동 최적화")
st.caption("빌드 버전: FULL-DATA-AUTODOWNLOAD-V2-(2) · 2026-11-19")

# Apply saved optimizer values before creating keyed widgets.
if st.session_state.get("apply_best_on_next_run"):
    res=st.session_state.get("optimization_result")
    if res:
        for key,value in res["best"].items(): st.session_state[f"param_{key}"]=float(value)
        st.session_state["parameters_applied"]=True
    st.session_state["apply_best_on_next_run"]=False

sb=st.sidebar
sb.header("운용 모드")
mode=sb.radio("데이터 입력 방식",["가상 파도 시뮬레이션","Akademik 전체 항해 데이터"],index=0,key="mode_select")
actual_mode=(mode=="Akademik 전체 항해 데이터")

# Persistent fixed/hardware + assumed sensor parameters
sb.header("안정화 장치 모델")
with sb.expander("서보 및 센서 고정 조건",expanded=False):
    tau=sb.slider("서보 응답 시간상수 τ(s)",0.02,0.4,0.10,0.01,key="fixed_tau")
    rate=sb.slider("서보 최대 각속도(°/s)",50,600,300,10,key="fixed_rate")
    limit=sb.slider("서보 회전 한계(±°)",5.0,30.0,15.0,1.0,key="fixed_limit")
    gyro_noise=sb.slider("가상 자이로 노이즈(°/s)",0.0,2.0,0.3,0.1,key="sensor_gyro_noise")
    gyro_bias=sb.slider("가상 자이로 바이어스(°/s)",0.0,2.0,0.3,0.1,key="sensor_gyro_bias")
    acc_noise=sb.slider("가속도계 환산각 노이즈(°)",0.0,3.0,0.5,0.1,key="sensor_acc_noise")
    accel_disturb=sb.slider("추가 가속도 오염 진폭(°)",0.0,5.0,0.0,0.1,key="sensor_acc_disturb")
sb.caption("센서 오차는 원본 ACE INS에서 직접 측정한 값이 아니라 시뮬레이터 가정값입니다.")

# Adjustable control params; can be filled by optimizer callback.
sb.header("제어 파라미터")
with sb.expander("FF + PD + 상보필터",expanded=True):
    ff=sb.slider("선행 보정 FF",0.0,1.2,1.0,0.01,key="param_ff")
    kp=sb.slider("P 이득 Kp",0.0,4.0,1.5,0.01,key="param_kp")
    kd=sb.slider("D 이득 Kd(s)",0.0,0.30,0.05,0.005,key="param_kd")
    alpha=sb.slider("상보필터 계수 α (20ms 기준)",0.90,0.999,0.99,0.001,key="param_alpha")

controller_on=sb.toggle("보정 작동",value=True,key="controller_on")

# Actual dataset mode: prominent main-page and sidebar download controls.
manifest=None; wave_df=None; archive_path=None; current_df=None; current_descriptor=None
if actual_mode:
    st.subheader("Akademik 전체 항해 데이터")
    st.info(
        "실제 INS 전체 항해 기록과 재구성 Hs/Tp 자료를 한 번에 자동 다운로드합니다. "
        "준비가 끝나면 날짜순 연속 구간을 자동 재생하고, 각 구간 종료 후 다음 기록으로 넘어갑니다."
    )
    ready_now = dataset_ready()
    with st.container(border=True):
        st.markdown("#### 전체 항해 데이터 자동 다운로드 및 등록")
        status_cols = st.columns(2)
        status_cols[0].metric("INS 원본 ZIP", "준비 완료" if ready_now else "미다운로드", "약 646.9 MB")
        status_cols[1].metric("파고·주기 CSV", "준비 완료" if ready_now else "미다운로드", "약 1.6 MB")
        if ready_now:
            st.success("이 실행 환경의 캐시에 전체 데이터가 있습니다. 아래에서 인덱스 상태를 확인합니다.")
            download_clicked = False
        else:
            st.warning("아직 이 실행 환경에 전체 데이터가 없습니다. 아래 버튼을 누르면 INS와 Hs/Tp 파일을 자동으로 받습니다.")
            main_download = st.button(
                "전체 항해 데이터 자동 다운로드 및 등록 시작",
                type="primary", width="stretch", key="download_full_archive_main",
            )
            side_download = sb.button(
                "전체 항해 데이터 다운로드 시작",
                type="primary", width="stretch", key="download_full_archive_sidebar",
            )
            download_clicked = bool(main_download or side_download)
        st.caption(
            "공식 출처: Zenodo ACE INS (646.9 MB) · Zenodo 재구성 Hs/Tp (1.6 MB). "
            "원본 파일은 GitHub에 올리지 않고 실행 환경 캐시에 저장합니다."
        )

    if not ready_now and download_clicked:
        try:
            with st.status("공개 ACE 데이터 다운로드 및 등록 중…", expanded=True) as status:
                status.write("① INS 전체 항해 ZIP을 다운로드하고 체크섬을 확인합니다.")
                archive_path, wave_path = ensure_dataset(status)
                status.write("② 날짜별 CSV를 읽어 시간 공백이 없는 연속 구간을 인덱싱합니다.")
                pbar = st.progress(0.0, text="항해 기록 인덱싱 시작")
                manifest = build_manifest(archive_path, pbar)
                wave_df = read_wave_csv(wave_path)
                status.update(
                    label=f"완료: {len(manifest):,}개 연속 구간 · 파랑 기록 {len(wave_df):,}개 등록",
                    state="complete", expanded=False,
                )
            st.success("전체 항해 데이터 준비가 끝났습니다. 화면을 새로 고친 뒤 자동 재생을 시작합니다.")
            st.rerun()
        except Exception as exc:
            st.error(f"데이터 자동 다운로드/등록에 실패했습니다: {type(exc).__name__}: {exc}")
            st.info("네트워크 연결 또는 실행 환경의 디스크 용량을 확인한 뒤 다시 시도해 주세요. 불완전한 임시 파일은 자동 삭제됩니다.")
            st.stop()

    if not ready_now:
        st.stop()

    archive_path=INS_ARCHIVE
    try:
        manifest=load_manifest(archive_path)
        wave_df=read_wave_csv(WAVE_CSV)
    except Exception as exc:
        st.error(f"저장된 데이터 인덱스를 읽지 못했습니다: {type(exc).__name__}: {exc}")
        if st.button("항해 데이터 인덱스 다시 만들기", key="rebuild_manifest"):
            MANIFEST_JSON.unlink(missing_ok=True)
            st.rerun()
        st.stop()
    st.success(
        f"전체 데이터 준비 완료 · 일별 파일 {len(set(x['member'] for x in manifest)):,}개 · "
        f"연속 재생 구간 {len(manifest):,}개 · 파랑 기록 {len(wave_df):,}개"
    )

    labels=[f"{i+1}/{len(manifest)} · {pd.Timestamp(d['start']).strftime('%Y-%m-%d %H:%M:%S')} UTC · {d['rows']:,} samples · {Path(d['member']).name}" for i,d in enumerate(manifest)]
    start_idx=sb.selectbox("처음 재생할 항해 구간",range(len(manifest)),format_func=lambda i:labels[i],index=0,key="actual_start_idx")
    auto_advance=sb.checkbox("구간 종료 시 다음 항해 기록 자동 재생",value=True,key="auto_advance")
    playback_speed=sb.slider("전체 항해 재생 배속",0.25,1000.0,20.0,0.25,key="actual_speed")
    refresh=sb.select_slider("화면 갱신 간격(s)",[0.15,0.25,0.4,0.6],value=0.25,key="refresh_actual")

    signature=f"actual|{INS_ARCHIVE.stat().st_size}|{len(manifest)}|{start_idx}"
    if st.session_state.get("source_signature")!=signature:
        st.session_state["source_signature"]=signature
        st.session_state["current_segment_idx"]=int(start_idx)
        st.session_state["current_actual_df"]=load_segment(archive_path,manifest[start_idx])
        first=st.session_state["current_actual_df"].iloc[0]
        st.session_state["sim"]=new_sim_state((first.ship_roll,first.ship_pitch))
        st.session_state["last_tick"]=time.perf_counter()
        st.session_state["actual_finished"]=False
    current_idx=int(st.session_state.get("current_segment_idx",start_idx))
    if "current_actual_df" not in st.session_state:
        st.session_state["current_actual_df"]=load_segment(archive_path,manifest[current_idx])
    current_df=st.session_state["current_actual_df"]
    current_descriptor=manifest[current_idx]
    duration=float(current_df["elapsed_s"].iloc[-1])
    sb.caption(f"현재 구간 {current_idx+1}/{len(manifest)} · {len(current_df):,}개 INS 기록 · 중앙 샘플 간격 {current_descriptor['median_dt']:.2f}s")
    if len(manifest)>1:
        sb.caption("자동 다음 구간 이동이 켜져 있으면 선택한 시작 구간부터 마지막 기록까지 순서대로 재생합니다.")

else:
    # Virtual conditions
    sb.header("가상 해양환경")
    sea_name=sb.selectbox("파도 상태",list(SEA_STATES.keys()),index=1,key="virtual_sea_state")
    sea=dict(SEA_STATES[sea_name])
    if sb.toggle("파도 조건 직접 변경",value=False,key="custom_wave"):
        sea["Hs"]=sb.slider("유의파고 Hs(m)",0.2,6.0,float(sea["Hs"]),0.1,key="virtual_hs")
        sea["Tp"]=sb.slider("피크 주기 Tp(s)",3.0,14.0,float(sea["Tp"]),0.5,key="virtual_tp")
        sea["direction"]=sb.slider("파도 방향(°)",0,360,int(sea["direction"]),5,key="virtual_dir")
    gain=sb.slider("흔들림 크기 배율",0.25,2.0,1.0,0.05,key="virtual_gain")
    comps=wave_components(sea["Tp"],sea["Hs"],sea["components"])
    playback_speed=sb.slider("재생 배속",0.25,20.0,1.0,0.25,key="virtual_speed")
    refresh=sb.select_slider("화면 갱신 간격(s)",[0.15,0.25,0.4,0.6],value=0.25,key="refresh_virtual")
    shock_roll=sb.button("Roll 충격",key="virtual_roll_shock")
    shock_pitch=sb.button("Pitch 충격",key="virtual_pitch_shock")
    signature=f"virtual|{sea_name}|{sea['Hs']}|{sea['Tp']}|{sea['direction']}|{gain}"
    if st.session_state.get("source_signature")!=signature:
        st.session_state["source_signature"]=signature
        st.session_state["sim"]=new_sim_state()
        st.session_state["last_tick"]=time.perf_counter()
    if "sim" not in st.session_state: st.session_state["sim"]=new_sim_state()
    if shock_roll: st.session_state["sim"]["shocks"].append({"t0":st.session_state["sim"]["t"],"axis":"roll","peak":sea["shock"]})
    if shock_pitch: st.session_state["sim"]["shocks"].append({"t0":st.session_state["sim"]["t"],"axis":"pitch","peak":0.7*sea["shock"]})
    st.session_state["actual_finished"]=False

# Reset button
if sb.button("현재 구간 / 가상 시험 처음부터",width="stretch",key="reset_sim"):
    if actual_mode and current_df is not None:
        first=current_df.iloc[0]
        st.session_state["sim"]=new_sim_state((first.ship_roll,first.ship_pitch))
        st.session_state["actual_finished"]=False
    else:
        st.session_state["sim"]=new_sim_state()
    st.session_state["last_tick"]=time.perf_counter()

# Main ship data and provenance
st.subheader("선박 기준")
cols=st.columns(5)
cols[0].metric("선박",SHIP["name"])
cols[1].metric("전장",f"{SHIP['length']:.2f} m")
cols[2].metric("선폭",f"{SHIP['breadth']:.2f} m")
cols[3].metric("배수량",f"{SHIP['displacement']:,} t")
cols[4].metric("흘수",f"{SHIP['draft']:.1f} m")
if actual_mode:
    st.info("실제 항해 모드: 선박 Roll/Pitch는 ACE INS 기록값입니다. IMU 측정·필터·서보·상판 응답은 캡스톤 장치의 수학적 모델이며 실제 하드웨어 측정값이 아닙니다.")
    st.caption("INS: Thomas & Pina Estany (2020), Zenodo DOI 10.5281/zenodo.3733228, CC BY 4.0. 공개 원자료는 보정되지 않았고 품질검사도 수행되지 않았습니다.")
    st.caption("Hs/Tp: Derkani et al. (2021), Zenodo DOI 10.5281/zenodo.4541564, CC BY 4.0. 선박 Heave 5분 구간과 RAO로 재구성한 해상상태 추정값입니다.")
else:
    code,label=classify_sea_state(sea["Hs"])
    st.caption(f"가상 조건: {sea_name} · Hs {sea['Hs']:.1f}m · Tp {sea['Tp']:.1f}s · WMO/Douglas {code} ({label})")

playing=sb.toggle("자동 재생",value=True,key="auto_play")

# Live simulation fragment
@st.fragment(run_every=refresh)
def live_panel():
    now=time.perf_counter()
    elapsed_wall=min(max(now-st.session_state.get("last_tick",now),0.0),1.0)
    st.session_state["last_tick"]=now
    if "sim" not in st.session_state:
        st.session_state["sim"]=new_sim_state()
    sim=st.session_state["sim"]
    if actual_mode:
        sim_df=st.session_state["current_actual_df"]
        seg_idx=int(st.session_state["current_segment_idx"])
        seg_desc=manifest[seg_idx]
        seg_duration=float(sim_df["elapsed_s"].iloc[-1])
    else:
        sim_df=None; seg_duration=float("inf"); seg_idx=0; seg_desc=None

    if playing and not st.session_state.get("actual_finished",False):
        n_steps=min(10000,max(1,int(round(elapsed_wall*playback_speed/CONTROL_DT))))
        for _ in range(n_steps):
            if actual_mode:
                sim_df=st.session_state["current_actual_df"]
                seg_idx=int(st.session_state["current_segment_idx"])
                seg_duration=float(sim_df["elapsed_s"].iloc[-1])
                if sim["t"] >= seg_duration:
                    if auto_advance and seg_idx+1 < len(manifest):
                        next_idx=seg_idx+1
                        try:
                            next_df=load_segment(archive_path,manifest[next_idx])
                        except Exception as exc:
                            st.error(f"다음 항해 구간 로딩 실패: {exc}")
                            st.session_state["actual_finished"]=True
                            break
                        st.session_state["current_segment_idx"]=next_idx
                        st.session_state["current_actual_df"]=next_df
                        f=next_df.iloc[0]
                        sim=new_sim_state((f.ship_roll,f.ship_pitch))
                        st.session_state["sim"]=sim
                        continue
                    else:
                        st.session_state["actual_finished"]=True
                        break
                motion=actual_motion_at(sim["t"],sim_df)
            else:
                motion=virtual_motion(sim["t"],comps,sea["direction"],sea["gain"]*gain,sim["shocks"])
            p={"ff":ff,"kp":kp,"kd":kd,"alpha":alpha,"tau":tau,"rate":float(rate),"limit":limit,
               "gyro_noise":gyro_noise,"gyro_bias":gyro_bias,"acc_noise":acc_noise}
            row=control_step(sim,motion,p,controller_on,accel_disturb if actual_mode else 0.0,CONTROL_DT)
            sim["history"].append(row)
            keep=int(HISTORY_S/CONTROL_DT)
            if len(sim["history"])>keep: del sim["history"][:-keep]
        st.session_state["sim"]=sim

    sim=st.session_state["sim"]
    if not sim["history"]:
        if actual_mode:
            sim_df=st.session_state["current_actual_df"]
            motion=actual_motion_at(0.0,sim_df)
        else:
            motion=virtual_motion(0.0,comps,sea["direction"],sea["gain"]*gain,sim["shocks"])
        p={"ff":ff,"kp":kp,"kd":kd,"alpha":alpha,"tau":tau,"rate":float(rate),"limit":limit,
           "gyro_noise":gyro_noise,"gyro_bias":gyro_bias,"acc_noise":acc_noise}
        sim["history"].append(control_step(sim,motion,p,controller_on,accel_disturb if actual_mode else 0.0,CONTROL_DT))
        st.session_state["sim"]=sim
    df=pd.DataFrame(sim["history"])
    row=df.iloc[-1]
    metric_cols=st.columns(5)
    metric_cols[0].metric("선박 Roll",f"{row.ship_roll:+.2f}°")
    metric_cols[1].metric("선박 Pitch",f"{row.ship_pitch:+.2f}°")
    metric_cols[2].metric("상판 Roll",f"{row.plate_roll:+.2f}°")
    metric_cols[3].metric("상판 Pitch",f"{row.plate_pitch:+.2f}°")
    if actual_mode:
        idx=int(st.session_state["current_segment_idx"])
        dnow=st.session_state["current_actual_df"]
        durnow=float(dnow["elapsed_s"].iloc[-1])
        metric_cols[4].metric("전체 항해 구간",f"{idx+1:,}/{len(manifest):,}")
        if row.timestamp is not None:
            h=row.heading if np.isfinite(row.heading) else np.nan
            ns=row.north_speed if np.isfinite(row.north_speed) else np.nan
            es=row.east_speed if np.isfinite(row.east_speed) else np.nan
            st.caption(f"현재 파일: {Path(manifest[idx]['member']).name} · INS 시각 {row.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC · 선수방향 {h:.1f}° · 북향속도 {ns:.2f}m/s · 동향속도 {es:.2f}m/s")
        wave=nearest_wave(row.timestamp,wave_df)
        if wave and np.isfinite(wave["Hs"]):
            code,label=classify_sea_state(wave["Hs"])
            tp=f"Tp {wave['Tp']:.2f}s" if np.isfinite(wave["Tp"]) else "Tp 자료 없음"
            st.markdown(f"**실제 항해 파랑 조건(재구성):** Hs {wave['Hs']:.2f}m · {tp} · WMO/Douglas {code} ({label})")
            st.caption(f"파랑 기준 시각 {wave['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC · INS 시각 차이 {wave['delta_s']:.0f}s · 최근접 매칭, 보간 없음")
        else:
            st.warning("현재 INS 시각과 180초 이내에 일치하는 Hs/Tp 기록이 없습니다.")
        st.caption(f"현재 구간 경과 {min(row.time,durnow):.1f}/{durnow:.1f}s · 자동 다음 구간 이동 {'ON' if auto_advance else 'OFF'} · 배속 {playback_speed:.2f}x")
    else:
        metric_cols[4].metric("진행 시간",f"{row.time:.1f}s")

    st.plotly_chart(scene_figure(row,actual_mode,comps if not actual_mode else None,sea["direction"] if not actual_mode else 0.0,controller_on),width="stretch",key="ship_scene_live")
    plotted=df.iloc[::2] if len(df)>1 else df
    c1,c2=st.columns(2)
    c1.plotly_chart(angle_figure(plotted,"roll","Roll: 선박 자세와 안정화 상판"),width="stretch",key="roll_live")
    c2.plotly_chart(angle_figure(plotted,"pitch","Pitch: 선박 자세와 안정화 상판"),width="stretch",key="pitch_live")
    a,b=st.columns([1.1,1])
    with a:
        st.markdown("**안정화 성능 지표 (최근 30초)**")
        st.dataframe(live_performance(df,limit),hide_index=True,width="stretch")
        if (df[["joint_roll","joint_pitch"]].abs()>=limit-1e-6).any().any():
            st.warning("서보 회전 한계에 도달했습니다. 이 구간에서는 상판이 수평을 유지하지 못할 수 있습니다.")
        if actual_mode:
            st.caption("주의: INS 원자료는 1Hz입니다. 실시간 화면은 50Hz 제어 모델 입력을 위해 선형 보간했으며 실제 고주파 흔들림을 복원하지 않습니다.")
        else:
            st.caption("가상 파도는 파도 경사 기반 단순화 모델이며 실선 RAO 모델이 아닙니다.")
    with b:
        st.plotly_chart(sensor_figure(plotted),width="stretch",key="sensor_live")
    st.download_button("현재 구간 시뮬레이션 CSV 저장",df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="stabilizer_current_segment.csv",mime="text/csv",key=f"dl_current_{len(df)}_{row.time:.2f}")
    if actual_mode and st.session_state.get("actual_finished",False):
        st.success("전체 항해 데이터의 마지막 연속 구간까지 재생했습니다.")

live_panel()

# -----------------------------------------------------------------------------
# Auto tuner UI (available in actual mode only)
# -----------------------------------------------------------------------------
if actual_mode:
    st.divider()
    st.header("실제 항해 데이터 기반 자동 제어 최적화")
    st.write("현재 준비된 전체 INS 일별 파일에서 여러 날짜·연속 구간의 대표 4분 창을 자동 추출합니다. FF, Kp, Kd, α 후보들을 반복 시뮬레이션하고 별도 날짜의 검증 데이터로 결과를 확인합니다.")
    st.warning("최적화는 1Hz 원자료 해상도에서 수행합니다. 50Hz로 보간한 가짜 추가 측정정보를 최적화 데이터인 것처럼 사용하지 않습니다. 결과는 시뮬레이터 추천값이며 실제 장치의 검증 완료값은 아닙니다.")
    c1,c2,c3=st.columns(3)
    c1.metric("튜닝 가능 일별 파일",f"{len(set(x['member'] for x in manifest)):,}")
    c2.metric("연속 구간",f"{len(manifest):,}")
    c3.metric("조정 파라미터","FF · Kp · Kd · α")
    trial_count=st.slider("자동 탐색 후보 수",24,80,40,4,key="tune_trials")
    fixed={"tau":tau,"rate":float(rate),"limit":limit,"gyro_noise":gyro_noise,
           "gyro_bias":gyro_bias,"acc_noise":acc_noise,"accel_disturb":accel_disturb}
    current={"ff":ff,"kp":kp,"kd":kd,"alpha":alpha}
    if st.button("전체 항해 데이터로 자동 최적값 찾기",type="primary",width="stretch"):
        try:
            with st.status("전체 항해 구간에서 튜닝 데이터 준비 중…",expanded=True) as status:
                status.write("모든 일별 파일에서 여러 날짜의 대표 4분 구간을 추출합니다.")
                train_windows,valid_windows,n_days,n_windows=make_tuning_windows(archive_path,manifest)
                status.write(f"준비 완료: {n_days}개 일별 파일 · {n_windows}개 4분 창 · 학습 {len(train_windows)}개 · 검증 {len(valid_windows)}개")
                bar=st.progress(0.0)
                def progress(frac,msg): bar.progress(min(frac,1.0),text=msg)
                result=optimize_params(train_windows,valid_windows,current,fixed,trial_count,progress)
                result["n_days"]=n_days; result["n_windows"]=n_windows
                st.session_state["optimization_result"]=result
                status.update(label="자동 최적화 완료",state="complete",expanded=False)
        except Exception as exc:
            st.error(f"자동 최적화 중 오류: {exc}")
    result=st.session_state.get("optimization_result")
    if result:
        st.subheader("자동 최적화 결과")
        best=result["best"]
        st.success("추천 파라미터 탐색이 완료됐습니다. 아래 값은 시뮬레이터의 현재 센서·서보 가정값에 종속됩니다.")
        mc=st.columns(4)
        mc[0].metric("추천 FF",f"{best['ff']:.3f}")
        mc[1].metric("추천 Kp",f"{best['kp']:.3f}")
        mc[2].metric("추천 Kd",f"{best['kd']:.3f}")
        mc[3].metric("추천 α",f"{best['alpha']:.4f}")
        comp=pd.DataFrame([
            {"검증 구분":"현재 파라미터","검증 점수":result["current_valid"]["score"],
             "Roll RMS":"%.3f°"%result["current_valid"]["roll_rms"],"Pitch RMS":"%.3f°"%result["current_valid"]["pitch_rms"],
             "Roll 최대각":"%.3f°"%result["current_valid"]["roll_max"],
             "Pitch 최대각":"%.3f°"%result["current_valid"]["pitch_max"],
             "서보 포화율":"%.2f%%"%(100*result["current_valid"]["saturation"])},
            {"검증 구분":"추천 파라미터","검증 점수":result["valid"]["score"],
             "Roll RMS":"%.3f°"%result["valid"]["roll_rms"],"Pitch RMS":"%.3f°"%result["valid"]["pitch_rms"],
             "Roll 최대각":"%.3f°"%result["valid"]["roll_max"],
             "Pitch 최대각":"%.3f°"%result["valid"]["pitch_max"],
             "서보 포화율":"%.2f%%"%(100*result["valid"]["saturation"])},
        ])
        st.markdown("**학습에 사용하지 않은 검증 구간 결과**")
        st.dataframe(comp,hide_index=True,width="stretch")
        cur=result["current_valid"]; opt=result["valid"]
        if cur["score"]>0:
            st.metric("검증 점수 개선율",f"{100*(cur['score']-opt['score'])/cur['score']:.1f}%")
        st.markdown("**상위 후보 10개**")
        st.dataframe(result["top"].round(4),hide_index=True,width="stretch")
        st.button("추천 파라미터를 제어 슬라이더에 적용",on_click=lambda: st.session_state.update({"apply_best_on_next_run":True}),
                  type="primary",width="stretch",key="apply_best_button")
        st.download_button("최적화 후보 및 결과 CSV 저장",result["top"].to_csv(index=False).encode("utf-8-sig"),
                           file_name="akademik_auto_tuning_candidates.csv",mime="text/csv")
        st.caption("서보 τ·최대 각속도·회전한계와 센서 노이즈/바이어스 설정은 최적화 중 고정했습니다. 이 고정값이 실제 하드웨어 사양과 측정치가 아니면 추천값의 신뢰도가 낮아집니다.")

with st.expander("모델·데이터 출처와 해석상의 한계"):
    st.markdown("""
**계산 흐름:** 선박 Roll/Pitch → 가상 IMU → 상보필터 → FF+PD → 서보 모델 → 상판 자세

**자동 데이터 관리**
- 첫 실행 때 Zenodo에서 INS ZIP(약 646.9MB)과 Hs/Tp CSV를 앱 캐시에 한 번 다운로드합니다.
- 원본 ZIP은 압축 해제하지 않고, 날짜별 파일을 ZIP 안에서 차례대로 읽습니다.
- 날짜순으로 연속 구간을 인덱싱하며, 구간 사이 긴 시간 공백을 가로질러 보간하지 않습니다.
- 현재 구간이 끝나면 자동으로 다음 날짜/구간으로 넘어갑니다.
- GitHub에는 대용량 원본 데이터를 올리지 않습니다.

**자동 최적화**
- FF, Kp, Kd, α만 조정합니다.
- τ, 최대 각속도, 기계적 회전한계, 가상 센서 노이즈/바이어스는 사용자가 정한 값으로 고정합니다.
- 모든 일별 파일에서 날짜별·구간별 대표 4분 창을 여러 개 추출합니다. 학습 구간에서 후보를 탐색하고, 일부 날짜는 검증용으로 분리합니다.
- 원자료는 1Hz이므로 자동 최적화는 1Hz 표본에서 수행합니다. 재생 화면만 50Hz 모델 입력을 위해 보간합니다.
- 목적함수는 Roll/Pitch RMS와 95백분위 잔차를 줄이고, 서보 사용량과 포화율을 억제하도록 구성했습니다.

**중요한 과학적 한계**
- 공개 INS는 ACE 기간 중 수집한 1Hz 항해 기록입니다. 제공기관은 원자료가 추가 보정되지 않았고 품질검사도 수행되지 않았다고 설명합니다.
- 기록 PC 시간이 GPS 및 다른 장비와 동기화되었는지는 확실하지 않다고 README에 기재되어 있습니다.
- Hs/Tp는 선박 Heave 운동의 5분 구간 스펙트럼과 선박별 RAO를 이용해 재구성한 자료입니다. 독립 부표의 동시 직접 측정값이 아닙니다.
- INS 자료에는 실제 제작할 캡스톤 플랫폼의 IMU 원시 자이로/가속도 신호가 들어 있지 않습니다. 센서 오차는 모델 가정값입니다.
- 본 프로그램의 서보 모델은 1차 지연·속도 제한·각도 제한을 단순화해 표현합니다. 실제 DS51150의 응답, 부하, 기어 유격, 전원 강하, 구조 탄성은 실측으로 검증해야 합니다.
- 따라서 자동 최적값은 **이 시뮬레이터 모델에 대한 후보값**입니다. 실물 적용 전에는 낮은 이득부터 단계적으로 시험하고, 독립된 실제 IMU·서보 시험 데이터로 재튜닝해야 합니다.

**출처 및 라이선스**
- Thomas & Pina Estany (2020), ACE INS, Zenodo DOI 10.5281/zenodo.3733228, CC BY 4.0.
- Derkani et al. (2021), ACE reconstructed sea state, Zenodo DOI 10.5281/zenodo.4541564, CC BY 4.0.
- [ACE INS 전체 원자료](https://zenodo.org/records/3733228) · [Hs/Tp 재구성 자료](https://zenodo.org/records/4541564) · [AARI 선박 정보](https://www.aari.ru/eng)
""")
