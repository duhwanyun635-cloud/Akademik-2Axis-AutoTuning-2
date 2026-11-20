"""Akademik Tryoshnikov 2-axis stabilization simulator + automatic tuning (2nd independent app).

Modes
- Virtual: synthetic wave -> simplified ship motion -> simulated IMU -> complementary filter
           -> FF+PD -> servo model -> stabilized plate.
- Actual ACE voyage: automatically downloads the complete public daily INS archive and the
  reconstructed sea-state CSV from Zenodo, indexes all daily files, then replays every continuous
  segment in chronological order without asking the user to upload files one by one.
- Auto tuning (simulation-based random search + local search, stages 0-5: seeds and convergence,
  all candidates and Pareto front, stability_check and constrained objective, Kp x Kd grid,
  WMO sea-state binned tuning with transfer/regret matrices): searches FF, Kp, Kd and
  complementary-filter alpha using representative windows sampled across the daily INS files.
  Since stage 0 the tuner evaluates candidates with the SAME 50 Hz loop, the same linear
  interpolation and the same noise sequence as live playback (see simulate_batch and
  consistency_check). Servo parameters and assumed sensor-error settings are held fixed.

Important limitations
- ACE INS is uncorrected / not quality checked by the provider and was not synchronized with other
  onboard instrumentation as far as known.
- ACE INS is 1 Hz. Playback and tuning both interpolate between samples to run the 50 Hz model;
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
from plotly.subplots import make_subplots
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


def epoch_seconds(series: pd.Series) -> np.ndarray:
    """UTC 시각 → epoch 초. pandas 버전·시간 해상도(ns/us)에 상관없이 같은 값을 준다.
    (astype("int64")/1e9는 pandas 3의 마이크로초 해상도에서 1000배 틀어진다.)"""
    base = pd.Timestamp("1970-01-01", tz="UTC")
    return ((series - base) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def split_continuous(df: pd.DataFrame):
    ts = epoch_seconds(df["timestamp"])
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
    sec = wave_df.attrs.get("_epoch")
    if sec is None or len(sec) != len(wave_df):
        sec = epoch_seconds(wave_df["timestamp"])
        wave_df.attrs["_epoch"] = sec
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
WMO_CODES = [  # (코드, Hs 하한, Hs 상한, 이름) — WMO Sea State Code(Douglas 척도)
    (0, 0.0, 0.0, "고요(유리면)"), (1, 0.0, 0.1, "고요(잔물결)"), (2, 0.1, 0.5, "평온"), (3, 0.5, 1.25, "약간 거침"),
    (4, 1.25, 2.5, "보통"), (5, 2.5, 4.0, "거침"), (6, 4.0, 6.0, "매우 거침"), (7, 6.0, 9.0, "높음"),
    (8, 9.0, 14.0, "매우 높음"), (9, 14.0, math.inf, "극심"),
]


def classify_sea_state(hs) -> tuple[int | None, str]:
    """유의파고 Hs(m) → WMO 해상상태 코드 0~9. 자료가 없으면 (None, "자료 없음")."""
    try:
        hs = float(hs)
    except (TypeError, ValueError):
        return None, "자료 없음"
    if not np.isfinite(hs) or hs < 0:
        return None, "자료 없음"
    if hs == 0:
        return 0, WMO_CODES[0][3]
    for code, lo, hi, name in WMO_CODES[1:]:
        if hs <= hi:
            return code, name
    return 9, WMO_CODES[-1][3]


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
# Auto tuning: simulation-based random search + local search, evaluated with the live 50 Hz loop
# -----------------------------------------------------------------------------
TUNE_KEYS = ["ff", "kp", "kd", "alpha"]
# Kd 상한: 4단계 Kp×Kd 격자에서 확인한 안정 경계(기본 서보 τ=0.10s에서 Kd≈0.07부터 진동) 바로 위까지만 탐색한다.
# 그 이상은 기본 조건에서도 리밋사이클이므로 탐색할 이유가 없다. 경계 근처는 로그 스케일 샘플링으로 촘촘히 본다.
KD_MAX = 0.12
TUNE_BOUNDS = {"ff": (0.0, 1.2), "kp": (0.0, 4.0), "kd": (0.0, KD_MAX), "alpha": (0.90, 0.999)}
KD_LOG_MIN = 0.002
CASES = {
    "① FF·Kp·Kd·α 전부 튜닝": {},
    "② FF=1 고정, PD+α만 튜닝": {"ff": 1.0},
    "③ FF=0 (PD만 사용), PD+α만 튜닝": {"ff": 0.0},
}
OBJ_WEIGHTED = "가중합 방식 (기존 방식, effort 재정의)"
OBJ_CONSTRAINED = "제약 조건 방식 (포화율 ≤1%, 리밋사이클 없음)"
OBJECTIVES = [OBJ_CONSTRAINED, OBJ_WEIGHTED]

SEARCH_WINDOWS = 30            # 탐색 단계에서 쓰는 학습창 수(무작위 추출, 시드 고정)
SUBSET_SEED = 20261120         # 탐색용 학습창 추출 시드(+ 탐색 시드)
NOISE_SEED_BASE = 2026         # 창 i의 기본 센서 노이즈 시드 = NOISE_SEED_BASE + i
VALID_NOISE_SEEDS = 3          # 검증 평가에 쓰는 노이즈 시드 수(창마다)
SEED_STRIDE = 7919             # 같은 창의 r번째 노이즈 시드 = 기본 시드 + SEED_STRIDE*r
CONSISTENCY_TOL = 0.05         # 실시간 재생과 튜닝 평가의 허용 차이(5%)
SERIES_BUDGET = 16_000_000     # p95 계산용 시계열 저장 상한(원소 수, 약 64 MB)

# 리밋사이클·고주파 명령 판정 (임계값은 기본 조건의 안정/진동 사례로 보정)
HF_CUTOFF_HZ = 2.0             # 이 주파수 이상을 고주파 명령으로 본다
LC_HF_RATIO = 0.30             # 관절각 분산 중 2Hz 이상 성분 비율이 이 값을 넘고
LC_HF_RMS = 0.30               # 2Hz 이상 성분 RMS가 이 값(°)을 넘으면 리밋사이클
LC_FLIP_HZ = 3.0               # 또는 관절 각속도 부호 변화로 본 진동 주파수가 이 값(Hz)을 넘으면 리밋사이클
FLIP_DEADBAND = 5.0            # 부호 변화 계산에서 무시할 작은 각속도(°/s, 노이즈 떨림 제외)
SAT_CONSTRAINT = 0.01          # 제약 조건 방식: 포화율 상한
PENALTY = 5.0                  # 제약 위반 페널티
STAB_GAIN_FACTORS = (1.5, 2.0)
STAB_TAU_FACTORS = (0.5, 1.5)
STAB_SAT_MAX = 0.05
STAB_TOP_K = 6                 # 안정성 검사를 할 상위 후보 수
STAB_WINDOWS = 8               # 안정성 검사에 쓰는 창 수


def prior_tilt_rms(seg: pd.DataFrame, start_idx: int) -> dict:
    """창 시작 직전 L분(2~5분) 동안의 선박 기울기 RMS(√(roll²+pitch²)). 하부 IMU가 측정할 수 있는 지표에 해당."""
    el = seg["elapsed_s"].to_numpy(dtype=float)
    r = seg["ship_roll"].to_numpy(dtype=float); p = seg["ship_pitch"].to_numpy(dtype=float)
    t0 = el[start_idx]
    out = {}
    for L in (2, 3, 4, 5):
        m = (el >= t0 - 60*L) & (el < t0)
        out[L] = float(np.sqrt(np.mean(r[m]**2 + p[m]**2))) if m.sum() >= 30*L else np.nan
    return out


def label_window(it: dict, wave_df: pd.DataFrame | None) -> None:
    """5-2: 창 시작·중앙·끝 시각의 Hs/Tp를 nearest_wave로 붙이고 WMO 코드를 정한다."""
    df = it["df"]
    ts = df["timestamp"]
    picks = [ts.iloc[0], ts.iloc[len(ts)//2], ts.iloc[-1]]
    found = [nearest_wave(x, wave_df) for x in picks]
    ok = [f for f in found if f is not None and np.isfinite(f["Hs"])]
    it["date"] = ts.iloc[0].strftime("%Y-%m-%d")
    it["start_time"] = ts.iloc[0].strftime("%H:%M:%S")
    it["roll_rms"] = float(np.sqrt(np.mean(df["ship_roll"].to_numpy(float)**2)))
    it["pitch_rms"] = float(np.sqrt(np.mean(df["ship_pitch"].to_numpy(float)**2)))
    it["wave_ok"] = bool(found[1] is not None and len(ok) >= 2)
    if it["wave_ok"]:
        hs = np.array([f["Hs"] for f in ok]); tp = np.array([f["Tp"] for f in ok], dtype=float)
        it["Hs"] = float(np.mean(hs)); it["Tp"] = float(np.nanmean(tp)) if np.isfinite(tp).any() else np.nan
        it["delta_max"] = float(max(f["delta_s"] for f in ok))
        codes = {classify_sea_state(h)[0] for h in hs}
        it["wmo"], it["wmo_name"] = classify_sea_state(it["Hs"])
        it["boundary"] = len(codes) > 1
    else:
        it["Hs"] = np.nan; it["Tp"] = np.nan; it["delta_max"] = np.nan
        it["wmo"], it["wmo_name"] = None, "파랑 자료 없음"; it["boundary"] = False


def make_tuning_windows(archive_path: Path, manifest: list[dict], wave_df: pd.DataFrame | None = None,
                        max_per_day=2, window_rows=240, max_total_windows=180) -> dict:
    """대표 4분 창을 날짜 전체에서 균등하게 추출한다.
    이전 버전은 앞 날짜부터 읽다가 max_total_windows에 도달하면 멈춰서 뒤쪽 날짜가 빠졌다.
    이제는 사용할 날짜 수(max_total_windows/max_per_day)만큼 전체 기간에서 균등 간격으로 고른다."""
    unique_members = sorted(set(d["member"] for d in manifest), key=lambda name: Path(name).name)
    member_rank = {member: i for i, member in enumerate(unique_members)}
    max_days = max(1, max_total_windows // max_per_day)
    if len(unique_members) > max_days:
        pick = sorted(set(np.linspace(0, len(unique_members)-1, max_days).round().astype(int)))
        chosen = [unique_members[i] for i in pick]
    else:
        chosen = list(unique_members)
    items = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in chosen:
            try:
                with zf.open(member, "r") as f:
                    daily = read_ins_csv(f)
            except ValueError:
                continue
            segments, _, _ = split_continuous(daily)
            candidates = []
            for seg in segments:
                n = len(seg)
                if n < 60:
                    continue
                length = min(window_rows, n)
                possible = sorted(set(int(x) for x in np.linspace(0, max(0, n - length), min(max_per_day, 3))))
                for start_idx in possible:
                    window = seg.iloc[start_idx:start_idx + length].copy().reset_index(drop=True)
                    if len(window) < 60:
                        continue
                    elapsed = window["elapsed_s"].to_numpy(dtype=float)
                    window["roll_rate"] = np.gradient(window["ship_roll"].to_numpy(dtype=float), elapsed)
                    window["pitch_rate"] = np.gradient(window["ship_pitch"].to_numpy(dtype=float), elapsed)
                    candidates.append((window, prior_tilt_rms(seg, start_idx)))
            if len(candidates) > max_per_day:
                sel = np.linspace(0, len(candidates) - 1, max_per_day).round().astype(int)
                candidates = [candidates[i] for i in sorted(set(sel))]
            is_validation = (member_rank[member] % 5 == 0)
            for window, prior in candidates:
                items.append({"df": window, "member": member, "validation": is_validation, "prior": prior})
    if not items:
        raise ValueError("자동 튜닝을 위한 1분 이상 연속 구간을 만들지 못했습니다.")
    n_val = sum(it["validation"] for it in items)
    if len(items) - n_val < 2 or n_val < 1:   # 작은 자료: 창 단위로 나눈다
        for i, it in enumerate(items):
            it["validation"] = (i % 5 == 0)
    for gi, it in enumerate(items):
        it["id"] = gi
        it["seed"] = NOISE_SEED_BASE + gi
        label_window(it, wave_df)
    return {"items": items, "days_used": len(set(it["member"] for it in items)), "days_chosen": len(chosen),
            "days_total": len(unique_members), "n_windows": len(items)}


def window_inputs(df: pd.DataFrame) -> dict:
    """튜닝 창을 실시간 재생과 똑같은 50 Hz 입력으로 만든다.
    - 시간축: live_panel처럼 t=0에서 시작해 CONTROL_DT를 계속 더하고, t < 구간 길이일 때만 스텝을 진행한다.
    - 보간: actual_motion_at과 같은 np.interp(선형 보간)를 roll, pitch, roll_rate, pitch_rate에 적용한다."""
    x = df["elapsed_s"].to_numpy(dtype=float)
    x = x - x[0]
    roll = df["ship_roll"].to_numpy(dtype=float)
    pitch = df["ship_pitch"].to_numpy(dtype=float)
    rr = df["roll_rate"].to_numpy(dtype=float) if "roll_rate" in df else np.gradient(roll, x)
    pr = df["pitch_rate"].to_numpy(dtype=float) if "pitch_rate" in df else np.gradient(pitch, x)
    duration = float(x[-1])
    ts = []
    t = 0.0
    while t < duration:
        ts.append(t)
        t += CONTROL_DT
    t = np.asarray(ts if ts else [0.0], dtype=float)

    def interp(v):
        good = np.isfinite(v)
        return np.interp(t, x[good], v[good]) if good.sum() >= 2 else np.full_like(t, np.nan)
    return {"t": t, "ship": np.stack([interp(roll), interp(pitch)]), "rate": np.stack([interp(rr), interp(pr)]),
            "init": (float(roll[0]), float(pitch[0])), "duration": duration}


def noise_for(seed: int, n: int):
    """new_sim_state(seed)와 control_step이 뽑는 난수를 같은 순서로 미리 만든다.
    bias[축][하부, 상판], z[스텝][축][gyro_b, acc_b, gyro_p, acc_p]"""
    rng = np.random.default_rng(seed)
    bias = np.array([rng.normal(0, 1) for _ in range(4)]).reshape(2, 2)
    z = rng.normal(0.0, 1.0, size=(n, 2, 4))
    return bias, z


def highpass_coeffs(fc: float, fs: float):
    """2차 버터워스 고역통과 필터 계수(쌍선형 변환)."""
    K = math.tan(math.pi * fc / fs)
    q = 1 / math.sqrt(2)
    norm = 1 / (1 + K/q + K*K)
    return norm, -2*norm, norm, 2*(K*K - 1)*norm, (1 - K/q + K*K)*norm


HP_COEF = highpass_coeffs(HF_CUTOFF_HZ, 1 / CONTROL_DT)


def simulate_batch(inputs: list[dict], seeds: list[int], cands: list[dict], fixed: dict, progress_cb=None,
                   keep_p95: bool = True) -> dict:
    """여러 후보 × 여러 창을 한 번에 50 Hz로 시뮬레이션한다.
    식은 control_step과 한 줄씩 같고, 노이즈 난수 순서도 같다(창마다 seeds[w]).
    후보 dict에 "tau"가 있으면 그 후보만 서보 시간상수를 바꿔 계산한다(안정성 검사용).
    반환: 지표 이름 -> (후보 C, 창 W) 배열"""
    W, C = len(inputs), len(cands)
    lens = np.array([inp["t"].size for inp in inputs])
    N = int(lens.max())
    ship = np.zeros((N, 2, W)); rate = np.zeros((N, 2, W)); tt = np.zeros((N, W)); valid = np.zeros((N, W), bool)
    noise = np.zeros((N, 2, 4, W)); bias = np.zeros((2, 2, W)); init = np.zeros((2, W)); acc_err = np.zeros((N, 2, W))
    for w, (inp, sd) in enumerate(zip(inputs, seeds)):
        n = int(lens[w])
        ship[:n, :, w] = inp["ship"].T; ship[n:, :, w] = inp["ship"][:, -1]
        rate[:n, :, w] = inp["rate"].T
        tt[:n, w] = inp["t"]; valid[:n, w] = True
        if "acc" in inp:            # 가상 파도: 선박 가속도에 의한 가속도계 오염각
            acc_err[:n, :, w] = inp["acc"].T
        b, z = noise_for(int(sd), n)
        bias[:, :, w] = b; noise[:n, :, :, w] = z
        init[:, w] = inp["init"]
    ad = float(fixed.get("accel_disturb", 0.0))
    if ad > 0:  # control_step과 같은 가속도 오염 식
        acc_err[:, 0] = acc_err[:, 0] + ad*np.sin(2*np.pi*0.23*tt)
        acc_err[:, 1] = acc_err[:, 1] + ad*np.sin(2*np.pi*0.19*tt+0.7)

    def col(k, default=None):
        return np.array([c.get(k, default) for c in cands], dtype=float)[None, :, None]
    ff, kp, kd, alpha = col("ff"), col("kp"), col("kd"), col("alpha")
    tau = col("tau", float(fixed["tau"]))
    dt = CONTROL_DT
    rlim, limit = float(fixed["rate"]), float(fixed["limit"])
    gn, gbias, an = float(fixed["gyro_noise"]), float(fixed["gyro_bias"]), float(fixed["acc_noise"])
    b0, b1, b2, a1, a2 = HP_COEF
    shape = (2, C, W)
    base_est = np.broadcast_to(init[:, None, :], shape).copy()
    plate_est = base_est.copy()
    joint = np.zeros(shape); jrate = np.zeros(shape)
    bb = bias[:, 0][:, None, :]; pb = bias[:, 1][:, None, :]
    ss = np.zeros(shape); mx = np.zeros(shape); sat = np.zeros((C, W))
    jsum = np.zeros(shape); jss = np.zeros(shape); jr_ss = np.zeros(shape); ja_ss = np.zeros(shape)
    hx1 = np.zeros(shape); hx2 = np.zeros(shape); hy1 = np.zeros(shape); hy2 = np.zeros(shape); hp_ss = np.zeros(shape)
    flips = np.zeros(shape); last_sgn = np.zeros(shape)
    series = np.empty((N, 2, C, W), dtype=np.float32) if keep_p95 else None
    for k in range(N):
        sh = ship[k][:, None, :]; sr = rate[k][:, None, :]; ae = acc_err[k][:, None, :]; z = noise[k]
        plate_before = sh + joint
        plate_rate = sr + jrate
        gyro_b = sr + gbias*bb + gn*z[:, 0][:, None, :]
        acc_b = sh + ae + an*z[:, 1][:, None, :]
        base_est = alpha*(base_est + gyro_b*dt) + (1-alpha)*acc_b
        gyro_p = plate_rate + gbias*pb + gn*z[:, 2][:, None, :]
        acc_p = plate_before + ae + an*z[:, 3][:, None, :]
        plate_est = alpha*(plate_est + gyro_p*dt) + (1-alpha)*acc_p
        cmd = np.clip(-ff*base_est - kp*plate_est - kd*gyro_p, -limit, limit)
        d = np.clip((cmd - joint)*dt/tau, -rlim*dt, rlim*dt)
        new_joint = np.clip(joint + d, -limit, limit)
        new_rate = (new_joint - joint)/dt
        jacc = (new_rate - jrate)/dt
        jrate = new_rate
        joint = new_joint
        plate = sh + joint
        m = valid[k][None, None, :]
        ss += np.where(m, plate*plate, 0.0)
        np.maximum(mx, np.where(m, np.abs(plate), 0.0), out=mx)
        sat += ((np.abs(joint) >= limit*0.999).any(axis=0) & valid[k][None, :])
        jsum += np.where(m, joint, 0.0); jss += np.where(m, joint*joint, 0.0)
        jr_ss += np.where(m, jrate*jrate, 0.0); ja_ss += np.where(m, jacc*jacc, 0.0)
        hy = b0*joint + b1*hx1 + b2*hx2 - a1*hy1 - a2*hy2
        hx2 = hx1; hx1 = joint; hy2 = hy1; hy1 = hy
        hp_ss += np.where(m, hy*hy, 0.0)
        sgn = np.where(np.abs(jrate) > FLIP_DEADBAND, np.sign(jrate), 0.0)
        flips += np.where(m & (sgn != 0) & (last_sgn != 0) & (sgn != last_sgn), 1.0, 0.0)
        last_sgn = np.where(sgn != 0, sgn, last_sgn)
        if keep_p95:
            series[k] = np.abs(plate)
        if progress_cb and k % 600 == 0:
            progress_cb(k / N)
    n = lens[None, None, :].astype(float)
    if keep_p95:
        p95 = np.empty(shape)
        for w in range(W):
            p95[:, :, w] = np.percentile(series[:lens[w], :, :, w], 95, axis=0)
    else:
        p95 = np.full(shape, np.nan)
    ship_rms = np.stack([np.sqrt(np.mean(inp["ship"]**2, axis=1)) for inp in inputs], axis=1)   # (2, W)
    ship_p95 = np.stack([np.percentile(np.abs(inp["ship"]), 95, axis=1) for inp in inputs], axis=1)
    ship_max = np.stack([np.max(np.abs(inp["ship"]), axis=1) for inp in inputs], axis=1)
    rms_out = np.sqrt(ss / n)
    srms, sp95, smax = (np.maximum(v, 0.1)[:, None, :] for v in (ship_rms, ship_p95, ship_max))
    roll_ratio, pitch_ratio = rms_out[0]/srms[0], rms_out[1]/srms[1]
    rms_ratio = 0.5*(roll_ratio + pitch_ratio)
    p95_ratio = 0.5*(p95[0]/sp95[0] + p95[1]/sp95[1])
    max_ratio = 0.5*(mx[0]/smax[0] + mx[1]/smax[1])
    # 3단계 effort 재정의: 필요한 관절 각도 자체가 아니라 '움직임의 격렬함'에 벌점을 준다.
    rate_rms = np.sqrt(jr_ss / n).mean(axis=0) / rlim                       # 관절 각속도 RMS / 최대 각속도
    acc_rms = np.sqrt(ja_ss / n).mean(axis=0) / (rlim / tau[0])             # 관절 각가속도 RMS / (최대 각속도/τ)
    var = np.maximum(jss/n - (jsum/n)**2, 1e-12)
    hf_axis = np.clip((hp_ss/n) / var, 0.0, 1.0)
    hf_rms_axis = np.sqrt(hp_ss/n)
    flip_axis = flips / (n*dt) / 2.0                                        # 부호 변화 2번 = 진동 1주기
    hf = hf_axis.max(axis=0)
    flip_hz = flip_axis.max(axis=0)
    lc = (((hf_axis > LC_HF_RATIO) & (hf_rms_axis > LC_HF_RMS)) | (flip_axis > LC_FLIP_HZ)).any(axis=0).astype(float)
    saturation = sat / n[0]
    score_w = (0.45*rms_ratio + 0.20*p95_ratio + 0.15*max_ratio
               + 0.05*rate_rms + 0.05*np.minimum(acc_rms, 1.0) + 0.05*hf + 0.05*saturation)
    score_c = rms_ratio + PENALTY*(saturation > SAT_CONSTRAINT) + PENALTY*lc
    rep_ = lambda v: np.broadcast_to(v[None, :], (C, W))
    return {"score_w": score_w, "score_c": score_c, "rms_ratio": rms_ratio,
            "roll_rms": rms_out[0], "pitch_rms": rms_out[1],
            "roll_rms_before": rep_(srms[0, 0]), "pitch_rms_before": rep_(srms[1, 0]),
            "roll_p95": p95[0], "pitch_p95": p95[1], "roll_max": mx[0], "pitch_max": mx[1],
            "roll_ratio": roll_ratio, "pitch_ratio": pitch_ratio,
            "rate_rms": rate_rms, "acc_rms": acc_rms, "hf": hf, "hf_rms": hf_rms_axis.max(axis=0), "flip_hz": flip_hz,
            "lc": lc, "saturation": saturation, "steps": rep_(lens.astype(float))}


METRIC_KEYS = ["score_w", "score_c", "rms_ratio", "roll_rms", "pitch_rms", "roll_rms_before", "pitch_rms_before",
               "roll_p95", "pitch_p95", "roll_max", "pitch_max", "roll_ratio", "pitch_ratio", "rate_rms", "acc_rms",
               "hf", "hf_rms", "flip_hz", "lc", "saturation"]


def as_item(x):
    return {"df": x} if isinstance(x, pd.DataFrame) else x


def prepare_inputs(items: list, n_seeds: int = 1):
    """창 목록을 50 Hz 입력과 노이즈 시드로 바꾼다. 창마다 n_seeds개의 노이즈 시드를 쓴다.
    반환: inputs, seeds, owners(입력 → 창 번호)"""
    inputs, seeds, owners = [], [], []
    for j, raw in enumerate(items):
        it = as_item(raw)
        inp = it if "t" in it else window_inputs(it["df"])
        base = int(it.get("seed", NOISE_SEED_BASE + j))
        for r in range(n_seeds):
            inputs.append(inp); seeds.append(base + SEED_STRIDE*r); owners.append(j)
    return inputs, seeds, np.array(owners, dtype=int)


def objective_key(objective: str) -> str:
    return "score_c" if objective == OBJ_CONSTRAINED else "score_w"


def evaluate(inputs, seeds, cands, fixed, objective=OBJ_CONSTRAINED, owners=None, groups=None,
             aggregate="windows", progress_cb=None) -> dict:
    """후보들을 평가한다.
    - owners: 같은 창의 여러 노이즈 시드를 창 단위로 평균하기 위한 번호
    - aggregate: "windows"(창 평균), "group_mean"(구간 평균의 평균), "group_max"(구간 평균의 최댓값)
    반환: {"summary": 후보별 평균 지표 목록, "per_item": 지표 -> (C, 창 수) 배열}"""
    keep = objective != OBJ_CONSTRAINED
    W = len(inputs)
    N = max(inp["t"].size for inp in inputs)
    chunk = max(1, int(SERIES_BUDGET // (2 * W * N))) if keep else max(1, 120_000 // W)
    parts = []
    for a in range(0, len(cands), chunk):
        part = cands[a:a + chunk]
        cb = (lambda f, a=a, k=len(part): progress_cb((a + f*k) / len(cands))) if progress_cb else None
        parts.append(simulate_batch(inputs, seeds, part, fixed, cb, keep_p95=keep))
    m = {k: np.concatenate([p[k] for p in parts], axis=0) for k in METRIC_KEYS}
    if owners is None:
        owners = np.arange(W)
    n_items = int(owners.max()) + 1
    per_item = {}
    for k in METRIC_KEYS:
        acc = np.zeros((len(cands), n_items)); cnt = np.bincount(owners, minlength=n_items).astype(float)
        for j in range(n_items):
            acc[:, j] = m[k][:, owners == j].mean(axis=1)
        per_item[k] = acc
    per_item["lc_any"] = np.zeros((len(cands), n_items))
    for j in range(n_items):
        per_item["lc_any"][:, j] = m["lc"][:, owners == j].max(axis=1)
    sk = objective_key(objective)
    score_items = per_item[sk]
    if aggregate == "windows" or groups is None:
        agg = score_items.mean(axis=1)
    else:
        groups = np.asarray(groups)
        gm = np.stack([score_items[:, groups == g].mean(axis=1) for g in np.unique(groups)], axis=1)
        agg = gm.mean(axis=1) if aggregate == "group_mean" else gm.max(axis=1)
    summary = []
    for c in range(len(cands)):
        s = {k: (float(np.nanmean(per_item[k][c])) if np.isfinite(per_item[k][c]).any() else float("nan")) for k in METRIC_KEYS}
        s["lc_frac"] = float(per_item["lc_any"][c].mean())
        s["score"] = float(agg[c])
        summary.append(s)
    if progress_cb:
        progress_cb(1.0)
    return {"summary": summary, "per_item": per_item}


def evaluate_candidate(windows, p, fixed, objective=OBJ_WEIGHTED):
    """이전 API 호환용: 창 목록과 후보 하나를 50 Hz로 평가한다."""
    if not len(windows):
        return {"score": float("inf")}
    inputs, seeds, owners = prepare_inputs(windows)
    return evaluate(inputs, seeds, [p], fixed, objective, owners)["summary"][0]


def stability_variants(params: dict, fixed: dict) -> list[tuple[str, dict]]:
    v = [("기본 조건", dict(params))]
    for g in STAB_GAIN_FACTORS:
        v.append((f"게인 ×{g:g} (Kp {params['kp']*g:.2f}, Kd {params['kd']*g:.3f})", dict(params, kp=params["kp"]*g, kd=params["kd"]*g)))
    for f in STAB_TAU_FACTORS:
        v.append((f"서보 τ ×{f:g} ({fixed['tau']*f:.3f}s)", dict(params, tau=fixed["tau"]*f)))
    return v


def stability_check_many(cands: list[dict], inputs, seeds, owners, fixed, progress_cb=None) -> list[dict]:
    """3단계 stability_check를 여러 후보에 한 번에 적용한다.
    통과 조건: 각 변형 조건에서 자세 RMS비 < 1.0, 포화율 < 5%, 기본 조건에서 리밋사이클 없음."""
    variants = [stability_variants(c, fixed) for c in cands]
    flat = [vp for vs in variants for _, vp in vs]
    res = evaluate(inputs, seeds, flat, fixed, OBJ_CONSTRAINED, owners, progress_cb=progress_cb)["summary"]
    out = []
    i = 0
    for c, vs in zip(cands, variants):
        rows = []
        for j, (name, _) in enumerate(vs):
            s = res[i]; i += 1
            ok = s["rms_ratio"] < 1.0 and s["saturation"] < STAB_SAT_MAX
            note = []
            if s["rms_ratio"] >= 1.0: note.append("RMS비 ≥ 1")
            if s["saturation"] >= STAB_SAT_MAX: note.append("포화율 ≥ 5%")
            if j == 0:
                if s["lc_frac"] > 0:
                    ok = False; note.append("리밋사이클 검출")
            rows.append({"조건": name, "RMS비": s["rms_ratio"], "포화율(%)": 100*s["saturation"],
                         "2Hz 이상 비율": s["hf"], "2Hz 이상 RMS(°)": s["hf_rms"], "부호변화 진동수(Hz)": s["flip_hz"],
                         "리밋사이클 창 비율": s["lc_frac"], "판정": "통과" if ok else "실패", "사유": ", ".join(note)})
        table = pd.DataFrame(rows)
        out.append({"passed": bool((table["판정"] == "통과").all()), "table": table, "base": res[i - len(vs)]})
    return out


def stability_check(params: dict, windows_subset: list, fixed: dict) -> dict:
    inputs, seeds, owners = prepare_inputs(windows_subset)
    return stability_check_many([params], inputs, seeds, owners, fixed)[0]


def sample_candidate(rng, case_fixed: dict, kd_log: bool) -> dict:
    c = {k: float(rng.uniform(*TUNE_BOUNDS[k])) for k in TUNE_BOUNDS}
    if kd_log:
        c["kd"] = 0.0 if rng.uniform() < 0.1 else float(10**rng.uniform(math.log10(KD_LOG_MIN), math.log10(KD_MAX)))
    c.update(case_fixed)
    return c


def perturb_candidate(rng, base: dict, case_fixed: dict, kd_log: bool) -> dict:
    c = {}
    for k, (lo, hi) in TUNE_BOUNDS.items():
        c[k] = float(np.clip(base[k] + rng.normal(0, (hi-lo)*0.08), lo, hi))
    if kd_log:
        kd0 = max(base["kd"], KD_LOG_MIN)
        c["kd"] = float(np.clip(math.exp(math.log(kd0) + rng.normal(0, 0.4)), 0.0, KD_MAX))
    c.update(case_fixed)
    return c


def stratified_subset(n_items: int, groups, n_pick: int, rng) -> list[int]:
    if n_items <= n_pick:
        return list(range(n_items))
    if groups is None:
        return sorted(int(i) for i in rng.choice(n_items, n_pick, replace=False))
    groups = np.asarray(groups)
    ug = np.unique(groups)
    per = max(1, n_pick // len(ug))
    pick = []
    for g in ug:
        idx = np.where(groups == g)[0]
        pick += [int(i) for i in rng.choice(idx, min(per, len(idx)), replace=False)]
    return sorted(pick)


def optimize_params(train_windows, valid_windows, current, fixed, n_trials=36, progress_cb=None,
                    n_search=SEARCH_WINDOWS, seed=42, objective=OBJ_CONSTRAINED, case_fixed=None, kd_log=True,
                    n_valid_seeds=VALID_NOISE_SEEDS, train_groups=None, aggregate="windows", do_stability=True):
    """시뮬레이션 기반 자동 튜닝(무작위 탐색 + 국소 탐색).
    - 탐색 단계: 학습창 중 n_search개(시드에 따라 정해지는 무작위)만 사용
    - 최종 검증: 검증창 전체 × 노이즈 시드 n_valid_seeds개 평균
    - 최종 추천값은 stability_check를 통과한 후보 중 점수가 가장 좋은 것
    - 모든 평가는 실시간 재생과 같은 50 Hz 루프(simulate_batch)"""
    case_fixed = dict(case_fixed or {})
    rng = np.random.default_rng(seed)
    sub_rng = np.random.default_rng(SUBSET_SEED + seed)
    search_idx = stratified_subset(len(train_windows), train_groups, n_search, sub_rng)
    search_items = [train_windows[i] for i in search_idx]
    s_groups = None if train_groups is None else np.asarray(train_groups)[search_idx]
    s_in, s_seed, s_own = prepare_inputs(search_items)
    v_in, v_seed, v_own = prepare_inputs(valid_windows, n_valid_seeds)

    start = dict(current); start.update(case_fixed)
    candidates = [start]
    n_random = max(15, n_trials-9)
    for _ in range(n_random):
        candidates.append(sample_candidate(rng, case_fixed, kd_log))
    total = len(candidates) + 8 + (STAB_TOP_K if do_stability else 0) + 2

    def stage(frac0, count, label):
        if not progress_cb:
            return None
        return lambda f: progress_cb(min((frac0 + f*count)/total, 1.0), f"{label}")
    ev = evaluate(s_in, s_seed, candidates, fixed, objective, s_own, s_groups, aggregate,
                  stage(0, len(candidates), f"전역 탐색 · 후보 {len(candidates)}개 · 학습창 {len(search_items)}개 · 50Hz"))
    records = [(m["score"], c, m, "전역 탐색") for c, m in zip(candidates, ev["summary"])]
    ranked = sorted(records, key=lambda x: x[0])
    local_candidates = [perturb_candidate(rng, base, case_fixed, kd_log) for _, base, _, _ in ranked[:4] for _ in range(2)]
    ev2 = evaluate(s_in, s_seed, local_candidates, fixed, objective, s_own, s_groups, aggregate,
                   stage(len(candidates), len(local_candidates), "상위 후보 국소 탐색"))
    records += [(m["score"], c, m, "국소 탐색") for c, m in zip(local_candidates, ev2["summary"])]
    history, best_so_far = [], float("inf")
    for i, (sc, _, _, _) in enumerate(records, 1):
        best_so_far = min(best_so_far, sc)
        history.append({"평가 횟수": i, "점수": sc, "최고 점수": best_so_far})
    ranked = sorted(records, key=lambda x: x[0])

    stab_results = {}
    best_rec, stable = ranked[0], None
    if do_stability:
        k_in, k_seed, k_own = prepare_inputs(search_items[:STAB_WINDOWS])
        top = ranked[:STAB_TOP_K]
        checks = stability_check_many([r[1] for r in top], k_in, k_seed, k_own, fixed,
                                      stage(len(candidates)+8, STAB_TOP_K, "상위 후보 안정성 검사"))
        for r, chk in zip(top, checks):
            stab_results[id(r[1])] = chk
        passing = [r for r in top if stab_results[id(r[1])]["passed"]]
        stable = bool(passing)
        best_rec = passing[0] if passing else ranked[0]
    best_score, best_params, train_metrics, _ = best_rec
    current_train = records[0][2]
    vres = evaluate(v_in, v_seed, [best_params, dict(current)], fixed, objective, v_own,
                    progress_cb=stage(total-2, 2, f"최종 검증 · 검증창 {len(valid_windows)}개 × 노이즈 시드 {n_valid_seeds}개"))
    best_valid, current_valid = vres["summary"]
    all_rows = []
    for i, (sc, c, m, phase) in enumerate(records, 1):
        chk = stab_results.get(id(c))
        all_rows.append({"평가 순서": i, "단계": phase, "FF": c["ff"], "Kp": c["kp"], "Kd": c["kd"], "α": c["alpha"],
                         "학습 점수": sc, "자세 RMS비": m["rms_ratio"], "Roll RMS(°)": m["roll_rms"], "Pitch RMS(°)": m["pitch_rms"],
                         "Roll 최대각(°)": m["roll_max"], "Pitch 최대각(°)": m["pitch_max"],
                         "서보 각속도 RMS(정규화)": m["rate_rms"], "서보 각가속도 RMS(정규화)": m["acc_rms"],
                         "2Hz 이상 비율": m["hf"], "부호변화 진동수(Hz)": m["flip_hz"], "리밋사이클 창 비율": m["lc_frac"],
                         "서보 포화율(%)": 100*m["saturation"],
                         "안정성 검사": "-" if chk is None else ("통과" if chk["passed"] else "실패"),
                         "추천": c is best_params})
    all_df = pd.DataFrame(all_rows)
    top_df = all_df.sort_values("학습 점수").head(10).reset_index(drop=True)
    top_df.insert(0, "순위", range(1, len(top_df)+1))
    usage = {"search_used": len(search_items), "train_total": len(train_windows),
             "valid_used": len(valid_windows), "valid_total": len(valid_windows), "valid_seeds": n_valid_seeds,
             "search_steps": int(sum(i["t"].size for i in s_in)),
             "valid_steps": int(sum(i["t"].size for i in v_in)),
             "search_evals": len(records), "valid_evals": 2,
             "stab_windows": min(STAB_WINDOWS, len(search_items)) if do_stability else 0}
    return {"best": {k: float(best_params[k]) for k in TUNE_KEYS}, "train": train_metrics, "valid": best_valid,
            "current_train": current_train, "current_valid": current_valid, "top": top_df, "all": all_df,
            "history": pd.DataFrame(history), "n_candidates": len(records), "usage": usage, "seed": seed,
            "objective": objective, "case_fixed": case_fixed, "kd_log": kd_log, "stable": stable,
            "stability": stab_results.get(id(best_params)), "fixed": dict(fixed)}


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """두 지표를 모두 작게 하는 비지배해(파레토 해)."""
    keep = np.ones(len(x), bool)
    for i in range(len(x)):
        dominated = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        if dominated.any():
            keep[i] = False
    return keep


def grid_scan(ff: float, alpha: float, kps, kds, items, fixed, objective, progress_cb=None) -> dict:
    """4단계: FF·α 고정, Kp×Kd 격자 평가 + 각 칸의 stability_check."""
    inputs, seeds, owners = prepare_inputs(items)
    cands = [{"ff": ff, "kp": float(a), "kd": float(b), "alpha": alpha} for a in kps for b in kds]
    checks = stability_check_many(cands, inputs, seeds, owners, fixed, progress_cb)
    shape = (len(kps), len(kds))
    ratio = np.array([c["base"]["rms_ratio"] for c in checks]).reshape(shape)
    passed = np.array([c["passed"] for c in checks]).reshape(shape)
    sat = np.array([c["base"]["saturation"] for c in checks]).reshape(shape)
    if objective == OBJ_CONSTRAINED:
        score = np.array([c["base"]["score_c"] for c in checks]).reshape(shape)
    else:
        ev = evaluate(inputs, seeds, cands, fixed, objective, owners)
        score = np.array([s["score"] for s in ev["summary"]]).reshape(shape)
    return {"kps": np.asarray(kps), "kds": np.asarray(kds), "ratio": ratio, "passed": passed, "sat": sat,
            "score": score, "ff": ff, "alpha": alpha, "n_windows": len(items)}


def sensitivity_curves(base: dict, items, fixed, objective, n=13, kd_max_view=0.30, progress_cb=None) -> pd.DataFrame:
    """4단계 1D 민감도: 나머지를 base로 고정하고 한 파라미터만 범위 전체에서 바꾼다(Kd는 슬라이더 최대 0.30까지)."""
    inputs, seeds, owners = prepare_inputs(items)
    ranges = {"ff": (0.0, 1.2), "kp": (0.0, 4.0), "kd": (0.0, kd_max_view), "alpha": (0.90, 0.999)}
    cands, meta = [], []
    for k, (lo, hi) in ranges.items():
        for v in np.linspace(lo, hi, n):
            c = dict(base); c[k] = float(v); cands.append(c); meta.append((k, float(v)))
    ev = evaluate(inputs, seeds, cands, fixed, objective, owners, progress_cb=progress_cb)["summary"]
    return pd.DataFrame([{"파라미터": k, "값": v, "점수": s["score"], "자세 RMS비": s["rms_ratio"],
                          "포화율(%)": 100*s["saturation"], "리밋사이클 창 비율": s["lc_frac"]}
                         for (k, v), s in zip(meta, ev)])


# -----------------------------------------------------------------------------
# 5단계: 해상상태별 튜닝, 전이 행렬, 공통 게인
# -----------------------------------------------------------------------------
TP_DEFAULT_EDGES = (6.0, 9.0, 12.0)
PRIOR_DEFAULT_EDGES = "0.5, 1.0, 2.0"


def wmo_label(code) -> str:
    if code is None:
        return "파랑 자료 없음"
    c = [w for w in WMO_CODES if w[0] == code][0]
    rng_txt = "Hs 0 m" if code == 0 else (f"Hs > {c[1]:g} m" if not np.isfinite(c[2]) else f"Hs {c[1]:g}~{c[2]:g} m")
    return f"WMO {code} {c[3]} ({rng_txt})"


def tp_bin(tp: float, edges) -> int | None:
    if tp is None or not np.isfinite(tp):
        return None
    return int(np.searchsorted(np.asarray(edges), tp, side="right"))


def tp_label(b: int, edges) -> str:
    e = list(edges)
    if b == 0: return f"Tp < {e[0]:g} s"
    if b == len(e): return f"Tp ≥ {e[-1]:g} s"
    return f"Tp {e[b-1]:g}~{e[b]:g} s"


def parse_edges(text: str) -> list[float]:
    vals = sorted({float(x) for x in re.split(r"[,\s]+", text.strip()) if x})
    if not vals:
        raise ValueError("구간 경계를 하나 이상 입력해 주세요.")
    return vals


def assign_bins(items: list[dict], criterion: str, tp_edges, prior_minutes: int, prior_edges) -> tuple[list, dict]:
    """5-3: 창마다 구간 키를 붙인다. 쓸 수 없는 창은 None.
    반환: 창별 키 목록, 키 -> {"order", "label", "short"}"""
    keys, info = [], {}
    for it in items:
        k = None
        if criterion.startswith("A"):
            if it.get("wave_ok") and it.get("wmo") is not None:
                k = ("WMO", it["wmo"])
                info[k] = {"order": (it["wmo"],), "label": wmo_label(it["wmo"]), "short": f"WMO {it['wmo']}", "codes": [it["wmo"]]}
        elif criterion.startswith("B"):
            b = tp_bin(it.get("Tp"), tp_edges) if it.get("wave_ok") else None
            if b is not None:
                k = ("TP", b)
                info[k] = {"order": (b,), "label": tp_label(b, tp_edges), "short": tp_label(b, tp_edges)}
        elif criterion.startswith("C"):
            b = tp_bin(it.get("Tp"), tp_edges) if it.get("wave_ok") else None
            if b is not None and it.get("wmo") is not None:
                k = ("WMOxTP", it["wmo"], b)
                info[k] = {"order": (it["wmo"], b), "label": f"WMO {it['wmo']} × {tp_label(b, tp_edges)}",
                           "short": f"W{it['wmo']}·{tp_label(b, tp_edges)}"}
        else:
            v = it.get("prior", {}).get(prior_minutes, np.nan)
            if np.isfinite(v):
                b = int(np.searchsorted(np.asarray(prior_edges), v, side="right"))
                e = prior_edges
                lab = (f"직전 {prior_minutes}분 기울기 RMS < {e[0]:g}°" if b == 0 else
                       f"직전 {prior_minutes}분 기울기 RMS ≥ {e[-1]:g}°" if b == len(e) else
                       f"직전 {prior_minutes}분 기울기 RMS {e[b-1]:g}~{e[b]:g}°")
                k = ("IMU", b)
                info[k] = {"order": (b,), "label": lab, "short": lab.replace(f"직전 {prior_minutes}분 기울기 RMS ", "")}
        keys.append(k)
    return keys, info


def merge_bins(keys: list, info: dict, items: list[dict], min_train: int, min_valid: int):
    """학습창/검증창이 부족한 구간을 순서상 인접한 구간과 합친다. 반환: 창별 그룹 번호, 그룹 정보, 병합 기록"""
    order = sorted(info, key=lambda k: info[k]["order"])
    groups = [[k] for k in order]

    def counts(g):
        tr = sum(1 for kk, it in zip(keys, items) if kk in g and not it["validation"])
        va = sum(1 for kk, it in zip(keys, items) if kk in g and it["validation"])
        return tr, va
    log = []
    while len(groups) > 1:
        bad = [i for i, g in enumerate(groups) if counts(g)[0] < min_train or counts(g)[1] < min_valid]
        if not bad:
            break
        i = bad[0]
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if sum(counts(groups[i-1])) <= sum(counts(groups[i+1])) else i + 1
        a, b = sorted((i, j))
        before = (group_label(groups[a], info), group_label(groups[b], info))
        groups[a] = groups[a] + groups[b]
        del groups[b]
        log.append(f"{before[0]} + {before[1]} → {group_label(groups[a], info)}")
    key_to_group = {k: gi for gi, g in enumerate(groups) for k in g}
    gid = [key_to_group.get(k) if k is not None else None for k in keys]
    ginfo = []
    for gi, g in enumerate(groups):
        tr, va = counts(g)
        ginfo.append({"id": gi, "keys": g, "label": group_label(g, info), "n_train": tr, "n_valid": va,
                      "enough": tr >= min_train and va >= min_valid})
    return gid, ginfo, log


def group_label(g: list, info: dict) -> str:
    if len(g) == 1:
        return info[g[0]]["label"]
    if all(k[0] == "WMO" for k in g):
        codes = sorted(k[1] for k in g)
        lo = [w for w in WMO_CODES if w[0] == codes[0]][0][1]
        hi = [w for w in WMO_CODES if w[0] == codes[-1]][0][2]
        hi_txt = f"> {lo:g}" if not np.isfinite(hi) else f"{lo:g}~{hi:g}"
        return f"WMO {'+'.join(str(c) for c in codes)} 병합 (Hs {hi_txt} m)"
    return " + ".join(info[k]["short"] for k in sorted(g, key=lambda k: info[k]["order"])) + " 병합"


def window_table(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"창": it["id"], "날짜": it.get("date", "-"), "시작 시각(UTC)": it.get("start_time", "-"),
                          "Hs(m)": it.get("Hs", np.nan), "Tp(s)": it.get("Tp", np.nan),
                          "WMO 코드": it.get("wmo"), "WMO 이름": it.get("wmo_name", "-"),
                          "매칭 시간차 최대(s)": it.get("delta_max", np.nan), "WMO 경계 걸침": it.get("boundary", False),
                          "구분": "검증" if it["validation"] else "학습",
                          "선박 Roll RMS(°)": it.get("roll_rms", np.nan), "선박 Pitch RMS(°)": it.get("pitch_rms", np.nan),
                          "직전 3분 기울기 RMS(°)": it.get("prior", {}).get(3, np.nan)} for it in items])


def run_bin_analysis(items: list[dict], gid: list, ginfo: list[dict], current: dict, fixed: dict, objective: str,
                     case_fixed: dict, kd_log: bool, n_trials: int, n_rep: int, seed: int, progress_cb=None) -> dict:
    """5-4, 5-5: 구간별 최적화 → 전이 행렬·regret → 공통 게인(평균형/최악형) 비교."""
    groups = [g for g in ginfo if g["enough"]]
    if len(groups) < 1:
        raise ValueError("학습창/검증창 최소 개수를 만족하는 구간이 없습니다.")
    gids = [g["id"] for g in groups]
    tr_of = {g: [it for it, k in zip(items, gid) if k == g and not it["validation"]] for g in gids}
    va_of = {g: [it for it, k in zip(items, gid) if k == g and it["validation"]] for g in gids}
    jobs = len(gids) * n_rep + 2 + 1
    done = [0]

    def sub_cb(label):
        def cb(f, msg=""):
            if progress_cb:
                progress_cb(min((done[0] + f) / jobs, 1.0), f"{label} · {msg}")
        return cb
    per_group = {}
    for g in gids:
        reps = []
        for r in range(n_rep):
            res = optimize_params(tr_of[g], va_of[g], current, fixed, n_trials, sub_cb(f"구간 '{ginfo[g]['label']}' 시드 {r+1}/{n_rep}"),
                                  seed=seed + r, objective=objective, case_fixed=case_fixed, kd_log=kd_log)
            reps.append(res); done[0] += 1
        per_group[g] = reps
    all_train = [it for g in gids for it in tr_of[g]]
    all_groups = [g for g in gids for _ in tr_of[g]]
    common = {}
    for name, agg in (("평균형 공통 게인", "group_mean"), ("최악형 공통 게인", "group_max")):
        all_valid = [it for g in gids for it in va_of[g]]
        res = optimize_params(all_train, all_valid, current, fixed, n_trials, sub_cb(name), seed=seed,
                              objective=objective, case_fixed=case_fixed, kd_log=kd_log,
                              train_groups=all_groups, aggregate=agg)
        common[name] = res; done[0] += 1
    # 전이 행렬: 모든 게인 × 모든 구간의 검증창(노이즈 시드 VALID_NOISE_SEEDS개)
    gains = [(f"g[{ginfo[g]['label']}]", per_group[g][0]["best"]) for g in gids]
    gains += [(n, common[n]["best"]) for n in common] + [("현재 슬라이더 값", dict(current))]
    v_items = [it for g in gids for it in va_of[g]]
    v_group = np.array([g for g in gids for _ in va_of[g]])
    v_in, v_seed, v_own = prepare_inputs(v_items, VALID_NOISE_SEEDS)
    ev = evaluate(v_in, v_seed, [c for _, c in gains], fixed, objective, v_own, progress_cb=sub_cb("전이 행렬 평가"))
    done[0] += 1
    sk = objective_key(objective)
    J = np.stack([ev["per_item"][sk][:, v_group == g].mean(axis=1) for g in gids], axis=1)        # (gain, group)
    R = np.stack([ev["per_item"]["rms_ratio"][:, v_group == g].mean(axis=1) for g in gids], axis=1)
    S = np.stack([ev["per_item"]["saturation"][:, v_group == g].mean(axis=1) for g in gids], axis=1)
    own = np.array([J[i, i] for i in range(len(gids))])
    regret = J / own[None, :] - 1.0
    rows = []
    for i, g in enumerate(gids):
        reps = per_group[g]
        b = np.array([[r["best"][k] for k in TUNE_KEYS] for r in reps])
        gw = [it for it, k in zip(items, gid) if k == g]
        hs = np.array([it["Hs"] for it in gw], float); tp = np.array([it["Tp"] for it in gw], float)
        rows.append({"구간": ginfo[g]["label"], "학습창": ginfo[g]["n_train"], "검증창": ginfo[g]["n_valid"],
                     "Hs 평균(m)": np.nanmean(hs) if np.isfinite(hs).any() else np.nan, "Hs 표준편차": np.nanstd(hs) if np.isfinite(hs).any() else np.nan,
                     "Tp 평균(s)": np.nanmean(tp) if np.isfinite(tp).any() else np.nan, "Tp 표준편차": np.nanstd(tp) if np.isfinite(tp).any() else np.nan,
                     "FF": reps[0]["best"]["ff"], "Kp": reps[0]["best"]["kp"], "Kd": reps[0]["best"]["kd"], "α": reps[0]["best"]["alpha"],
                     "FF 시드 표준편차": b[:, 0].std(), "Kp 시드 표준편차": b[:, 1].std(), "Kd 시드 표준편차": b[:, 2].std(), "α 시드 표준편차": b[:, 3].std(),
                     "검증 RMS비": R[i, i], "포화율(%)": 100*S[i, i],
                     "안정성 검사": "통과" if reps[0]["stable"] else "실패(통과 후보 없음)"})
    labels = [ginfo[g]["label"] for g in gids]
    comp = []
    n_g = len(gids)
    for j, g in enumerate(gids):
        row = {"구간": labels[j]}
        for name, gi in (("구간별 게인", j), ("평균형 공통", n_g), ("최악형 공통", n_g + 1), ("현재 슬라이더 값", n_g + 2)):
            row[f"{name} RMS비"] = R[gi, j]; row[f"{name} 포화율(%)"] = 100*S[gi, j]; row[f"{name} regret(%)"] = 100*regret[gi, j]
        comp.append(row)
    comp = pd.DataFrame(comp)
    summary = []
    for name, gi in (("평균형", n_g), ("최악형", n_g + 1), ("현재 슬라이더 값", n_g + 2)):
        j = int(np.argmax(regret[gi])); prefix = "공통 게인" if name != "현재 슬라이더 값" else ""
        summary.append(f"{prefix}({name}) 사용 시 최대 regret {100*regret[gi, j]:.1f}% ({labels[j]})" if prefix else
                       f"현재 슬라이더 값 사용 시 최대 regret {100*regret[gi, j]:.1f}% ({labels[j]})")
    return {"labels": labels, "gain_names": [n for n, _ in gains], "J": J, "R": R, "S": S, "regret": regret,
            "table": pd.DataFrame(rows), "comparison": comp, "summary": summary,
            "common": {n: common[n]["best"] for n in common}, "objective": objective}


# -----------------------------------------------------------------------------
# 5-6: 가상 파도 대조 실험 (가상 파도는 실선 RAO 모델이 아님)
# -----------------------------------------------------------------------------
VIRTUAL_HS = (0.9, 1.9, 3.2, 5.0)     # WMO 3~6 대표값
VIRTUAL_TP = (6.0, 8.0, 10.0, 12.0)


def virtual_gain_for(hs: float) -> float:
    """기존 가상 모드 프리셋(Hs 0.5/2/4 m → 배율 0.6/2.0/3.5)을 Hs에 대해 선형 보간·외삽한 흔들림 배율."""
    xs, ys = (0.5, 2.0, 4.0), (0.6, 2.0, 3.5)
    if hs > xs[-1]:
        return ys[-1] + (hs - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(np.interp(hs, xs, ys))


def virtual_window(hs: float, tp: float, realization: int, duration: float, gain_mult: float, direction: float = 25.0) -> dict:
    """virtual_motion과 같은 식을 시간 배열로 계산한 가상 창(충격 없음)."""
    comps = wave_components(tp, hs, 5, seed=123 + 31*realization)
    n = int(round(duration / CONTROL_DT))
    t = np.arange(n) * CONTROL_DT
    freqs, amps, phases, spread = comps
    th = np.deg2rad(direction) + spread
    w = 2*np.pi*freqs; k = w*w/G
    ang = -w[None, :]*t[:, None] + phases[None, :]
    slope = -amps*k*np.sin(ang); slope_rate = amps*k*w*np.cos(ang)
    scale = np.rad2deg(0.75) * virtual_gain_for(hs) * gain_mult
    roll = (slope*np.sin(th)).sum(1)*scale; pitch = (slope*np.cos(th)).sum(1)*scale
    rr = (slope_rate*np.sin(th)).sum(1)*scale; pr = (slope_rate*np.cos(th)).sum(1)*scale
    acc = amps*w*w*np.sin(ang)
    ax = 0.5*(acc*np.cos(th)).sum(1); ay = 0.5*(acc*np.sin(th)).sum(1)
    acc_err = np.stack([np.rad2deg(np.arctan(ay/G)), -np.rad2deg(np.arctan(ax/G))])
    return {"t": t, "ship": np.stack([roll, pitch]), "rate": np.stack([rr, pr]), "acc": acc_err,
            "init": (float(roll[0]), float(pitch[0])), "duration": float(t[-1])}


def run_virtual_analysis(current: dict, fixed: dict, objective: str, case_fixed: dict, kd_log: bool, n_trials: int,
                         n_train: int, duration: float, gain_mult: float, seed: int, progress_cb=None) -> dict:
    conds = [(hs, tp) for hs in VIRTUAL_HS for tp in VIRTUAL_TP]
    fixed_v = dict(fixed, accel_disturb=0.0)   # 실시간 가상 모드도 추가 가속도 오염은 쓰지 않는다
    train_of, valid_of = {}, {}
    for ci, (hs, tp) in enumerate(conds):
        train_of[ci] = [dict(virtual_window(hs, tp, r, duration, gain_mult), seed=NOISE_SEED_BASE + 100*ci + r) for r in range(n_train)]
        valid_of[ci] = [dict(virtual_window(hs, tp, 50, duration, gain_mult), seed=NOISE_SEED_BASE + 100*ci + 50)]
    best = {}
    for ci, (hs, tp) in enumerate(conds):
        cb = (lambda f, m, ci=ci: progress_cb((ci + f) / (len(conds) + 1), f"Hs {conds[ci][0]} m · Tp {conds[ci][1]:g} s · {m}")) if progress_cb else None
        res = optimize_params(train_of[ci], valid_of[ci], current, fixed_v, n_trials, cb, seed=seed, objective=objective,
                              case_fixed=case_fixed, kd_log=kd_log, n_valid_seeds=1)
        best[ci] = res
    v_items = [valid_of[ci][0] for ci in range(len(conds))]
    v_in, v_seed, v_own = prepare_inputs(v_items, VALID_NOISE_SEEDS)
    ev = evaluate(v_in, v_seed, [best[ci]["best"] for ci in range(len(conds))], fixed_v, objective, v_own,
                  progress_cb=(lambda f: progress_cb((len(conds) + f) / (len(conds) + 1), "전이 행렬 평가")) if progress_cb else None)
    J = ev["per_item"][objective_key(objective)]                # (gain i, condition j)
    regret = J / np.diag(J)[None, :] - 1.0
    rows = []
    for ci, (hs, tp) in enumerate(conds):
        b = best[ci]["best"]; c, name = classify_sea_state(hs)
        rows.append({"Hs(m)": hs, "Tp(s)": tp, "WMO": f"{c} {name}", "FF": b["ff"], "Kp": b["kp"], "Kd": b["kd"], "α": b["alpha"],
                     "검증 RMS비": ev["per_item"]["rms_ratio"][ci, ci], "포화율(%)": 100*ev["per_item"]["saturation"][ci, ci],
                     "안정성 검사": "통과" if best[ci]["stable"] else "실패"})
    table = pd.DataFrame(rows)
    same_hs = [regret[i, j] for i, (h1, t1) in enumerate(conds) for j, (h2, t2) in enumerate(conds) if h1 == h2 and t1 != t2]
    same_tp = [regret[i, j] for i, (h1, t1) in enumerate(conds) for j, (h2, t2) in enumerate(conds) if t1 == t2 and h1 != h2]
    infl = []
    for k, span in (("FF", 1.2), ("Kp", 4.0), ("Kd", KD_MAX), ("α", 0.099)):
        piv = table.pivot(index="Hs(m)", columns="Tp(s)", values=k).to_numpy()
        infl.append({"파라미터": k, "Hs 고정·Tp 변화 시 표준편차(범위 대비 %)": 100*np.mean(piv.std(axis=1))/span,
                     "Tp 고정·Hs 변화 시 표준편차(범위 대비 %)": 100*np.mean(piv.std(axis=0))/span})
    infl = pd.DataFrame(infl)
    m_hs, m_tp = 100*float(np.mean(same_hs)), 100*float(np.mean(same_tp))
    verdict = ("Tp(주기)" if m_hs > m_tp else "Hs(파고)")
    summary = (f"같은 Hs에서 Tp가 다른 조건의 게인을 쓰면 평균 regret {m_hs:.1f}%, 같은 Tp에서 Hs가 다른 조건의 게인을 쓰면 평균 regret {m_tp:.1f}% → "
               f"이 가상 모델에서는 {verdict} 쪽이 최적 게인에 더 큰 영향을 줍니다.")
    return {"conds": conds, "table": table, "J": J, "regret": regret, "influence": infl, "summary": summary,
            "regret_same_hs": m_hs, "regret_same_tp": m_tp}


def consistency_check(segment: pd.DataFrame, params: dict, fixed: dict, seed: int = NOISE_SEED_BASE) -> pd.DataFrame:
    """같은 구간·같은 파라미터·같은 노이즈 시드에서
    (1) 자동 튜닝 평가(simulate_batch) RMS와 (2) 실시간 재생(actual_motion_at + control_step, live_panel과 같은 루프) RMS를 비교한다.
    RMS 계산식은 live_performance와 같다: sqrt(mean(x*x))."""
    seg = segment.copy().reset_index(drop=True)
    el = seg["elapsed_s"].to_numpy(dtype=float)
    seg["elapsed_s"] = el - el[0]
    for c in ("heading", "north_speed", "east_speed"):
        if c not in seg:
            seg[c] = np.nan
    if "timestamp" not in seg:
        seg["timestamp"] = None
    if "roll_rate" not in seg:
        seg["roll_rate"] = np.gradient(seg["ship_roll"].to_numpy(float), seg["elapsed_s"].to_numpy(float))
        seg["pitch_rate"] = np.gradient(seg["ship_pitch"].to_numpy(float), seg["elapsed_s"].to_numpy(float))
    inp = window_inputs(seg)
    m = simulate_batch([inp], [seed], [params], fixed)
    sim = new_sim_state((seg["ship_roll"].iloc[0], seg["ship_pitch"].iloc[0]), seed=seed)
    duration = float(seg["elapsed_s"].iloc[-1])
    p = {"ff": params["ff"], "kp": params["kp"], "kd": params["kd"], "alpha": params["alpha"],
         "tau": fixed["tau"], "rate": float(fixed["rate"]), "limit": fixed["limit"],
         "gyro_noise": fixed["gyro_noise"], "gyro_bias": fixed["gyro_bias"], "acc_noise": fixed["acc_noise"]}
    rows = []
    while sim["t"] < duration:
        motion = actual_motion_at(sim["t"], seg)
        rows.append(control_step(sim, motion, p, True, fixed.get("accel_disturb", 0.0), CONTROL_DT))
    live = pd.DataFrame(rows)
    out = []
    for axis, label in (("roll", "Roll"), ("pitch", "Pitch")):
        s_ = live[f"ship_{axis}"].to_numpy(float); p_ = live[f"plate_{axis}"].to_numpy(float)
        live_rms = float(np.sqrt(np.mean(p_*p_))); live_ship = float(np.sqrt(np.mean(s_*s_)))
        tune_rms = float(m[f"{axis}_rms"][0, 0])
        diff = abs(tune_rms - live_rms) / max(live_rms, 1e-12)
        out.append({"축": label, "스텝 수(튜닝/실시간)": f"{int(m['steps'][0, 0]):,} / {len(live):,}",
                    "보정 전 RMS(°)": live_ship, "튜닝 평가 RMS(°)": tune_rms, "실시간 재생 RMS(°)": live_rms,
                    "튜닝 평가 RMS비": tune_rms / max(live_ship, 1e-12), "실시간 RMS비": live_rms / max(live_ship, 1e-12),
                    "차이(%)": 100*diff, "판정(≤5%)": "통과" if diff <= CONSISTENCY_TOL else "실패"})
    return pd.DataFrame(out)


def consistency_windows(seg: pd.DataFrame, count: int, window_rows: int = 240) -> list[pd.DataFrame]:
    """현재 재생 구간에서 튜닝 창과 같은 방식(최대 240개 기록, 속도는 창 안에서 다시 계산)으로 검증 창을 뽑는다."""
    n = len(seg)
    if n < 60:
        return []
    length = min(window_rows, n)
    starts = sorted(set(int(x) for x in np.linspace(0, n - length, count)))
    out = []
    for a in starts:
        w = seg.iloc[a:a + length].copy().reset_index(drop=True)
        el = w["elapsed_s"].to_numpy(float)
        w["roll_rate"] = np.gradient(w["ship_roll"].to_numpy(float), el)
        w["pitch_rate"] = np.gradient(w["ship_pitch"].to_numpy(float), el)
        out.append(w)
    return out


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
st.caption("빌드 버전: FULL-DATA-AUTODOWNLOAD-V2-(2) · STAGE5-SEASTATE-TUNING · 2026-11-20")

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
    st.caption(f"가상 조건: {sea_name} · Hs {sea['Hs']:.1f}m · Tp {sea['Tp']:.1f}s · WMO {code if code is not None else '-'} ({label})")

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
            st.markdown(f"**실제 항해 파랑 조건(재구성):** Hs {wave['Hs']:.2f}m · {tp} · WMO {code if code is not None else '-'} ({label})")
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
fixed={"tau":tau,"rate":float(rate),"limit":limit,"gyro_noise":gyro_noise,
       "gyro_bias":gyro_bias,"acc_noise":acc_noise,"accel_disturb":accel_disturb}
current={"ff":ff,"kp":kp,"kd":kd,"alpha":alpha}
HS_NOTE=("Hs/Tp는 Derkani et al.(2021)이 선박 Heave 5분 구간 스펙트럼과 RAO로 재구성한 추정값이며, "
         "같은 시각 부표가 직접 측정한 값이 아닙니다.")


def fmt_params(p):
    return f"FF {p['ff']:.3f} · Kp {p['kp']:.3f} · Kd {p['kd']:.4f} · α {p['alpha']:.4f}"


def tuning_options(prefix: str, expanded=False):
    """자동 튜닝 공통 옵션(실험 케이스, 목적함수, 시드, Kd 로그 샘플링, 검증 노이즈 시드 수)."""
    with st.expander("자동 튜닝 옵션", expanded=expanded):
        o1,o2=st.columns(2)
        case=o1.selectbox("실험 케이스",list(CASES.keys()),index=0,key=f"{prefix}_case")
        objective=o2.selectbox("목적함수",OBJECTIVES,index=0,key=f"{prefix}_objective")
        o3,o4,o5=st.columns(3)
        seed=int(o3.number_input("탐색 시드",0,99999,42,1,key=f"{prefix}_seed"))
        kd_log=o4.toggle("Kd 로그 스케일 샘플링",value=True,key=f"{prefix}_kdlog")
        n_vs=int(o5.slider("검증 노이즈 시드 수",3,6,VALID_NOISE_SEEDS,1,key=f"{prefix}_vseeds"))
        st.caption(f"탐색 범위: FF {TUNE_BOUNDS['ff']}, Kp {TUNE_BOUNDS['kp']}, Kd {TUNE_BOUNDS['kd']}, α {TUNE_BOUNDS['alpha']}. "
                   f"Kd 상한 {KD_MAX}는 ④ Kp×Kd 격자에서 확인한 안정 경계(기본 τ에서 약 0.07~0.08) 바로 위입니다. "
                   "제약 조건 방식은 자세 RMS비를 최소화하고 포화율 >1% 또는 리밋사이클이면 큰 페널티를 줍니다. "
                   "가중합 방식은 0.45·RMS비 + 0.20·95% 잔차비 + 0.15·최대각비 + 0.05·(각속도 RMS + 각가속도 RMS + 2Hz 이상 비율 + 포화율)입니다.")
    return {"case":case,"case_fixed":CASES[case],"objective":objective,"seed":seed,"kd_log":kd_log,"n_valid_seeds":n_vs}


def heatmap(z, x, y, title, zlabel, zmin=None, zmax=None, text_fmt=".2f", colorscale="Viridis_r"):
    fig=go.Figure(go.Heatmap(z=z,x=x,y=y,colorscale=colorscale,zmin=zmin,zmax=zmax,colorbar=dict(title=zlabel),
                             text=np.vectorize(lambda v: format(v,text_fmt))(z),texttemplate="%{text}",hoverongaps=False))
    fig.update_layout(title=title,height=max(320,60+36*len(y)),margin=dict(l=10,r=10,t=40,b=10))
    return fig


if actual_mode:
    st.divider()
    st.header("시뮬레이션 기반 자동 튜닝(무작위 탐색 + 국소 탐색)")
    st.write("현재 준비된 전체 INS 일별 파일에서 여러 날짜·연속 구간의 대표 4분 창을 자동 추출합니다. FF, Kp, Kd, α 후보들을 반복 시뮬레이션하고 별도 날짜의 검증 데이터로 결과를 확인합니다.")
    st.warning("자동 튜닝 평가는 실시간 재생과 같은 50Hz 제어 루프, 같은 선형 보간, 같은 센서 노이즈 생성 순서를 사용합니다. "
               "입력은 1Hz 원자료를 보간한 값이므로 기록되지 않은 고주파 운동은 포함되지 않습니다. 결과는 시뮬레이터 추천값이며 실제 장치의 검증 완료값은 아닙니다.")
    st.caption(f"계산량 때문에 탐색 단계는 학습창 중 최대 {SEARCH_WINDOWS}개(탐색 시드로 정해지는 무작위 추출)만 쓰고, 최종 검증만 검증창 전체 × 노이즈 시드 여러 개를 씁니다.")
    c1,c2,c3=st.columns(3)
    c1.metric("튜닝 가능 일별 파일",f"{len(set(x['member'] for x in manifest)):,}")
    c2.metric("연속 구간",f"{len(manifest):,}")
    c3.metric("조정 파라미터","FF · Kp · Kd · α")
    trial_count=st.slider("자동 탐색 후보 수",24,80,40,4,key="tune_trials")
    opts=tuning_options("main")

    def get_pack():
        key=f"{INS_ARCHIVE.stat().st_size}|{len(manifest)}|{len(wave_df)}"
        if st.session_state.get("tune_pack_key")!=key:
            with st.status("전체 항해 기간에서 튜닝 창 준비 중…",expanded=True) as status:
                status.write("일별 파일을 전체 기간에서 균등하게 골라 대표 4분 창을 추출하고 파랑 라벨을 붙입니다.")
                pack=make_tuning_windows(archive_path,manifest,wave_df)
                status.update(label=f"튜닝 창 {pack['n_windows']}개 준비 완료",state="complete",expanded=False)
            st.session_state["tune_pack"]=pack; st.session_state["tune_pack_key"]=key
        return st.session_state["tune_pack"]

    def split_items(pack):
        tr=[it for it in pack["items"] if not it["validation"]]
        va=[it for it in pack["items"] if it["validation"]]
        return tr,va

    def run_opt(pack, o, seed, progress=None):
        tr,va=split_items(pack)
        return optimize_params(tr,va,current,fixed,trial_count,progress,seed=seed,objective=o["objective"],
                               case_fixed=o["case_fixed"],kd_log=o["kd_log"],n_valid_seeds=o["n_valid_seeds"])

    if st.button("전체 항해 데이터로 자동 최적값 찾기",type="primary",width="stretch"):
        try:
            pack=get_pack()
            tr,va=split_items(pack)
            st.write(f"준비 완료: 일별 파일 {pack['days_used']}/{pack['days_total']}일 사용 · 4분 창 {pack['n_windows']}개 · 학습 {len(tr)}개 · 검증 {len(va)}개")
            st.write(f"탐색 단계: 학습창 {min(SEARCH_WINDOWS,len(tr))}/{len(tr)}개 · 최종 검증: 검증창 {len(va)}/{len(va)}개 × 노이즈 시드 {opts['n_valid_seeds']}개 · 50Hz 평가")
            bar=st.progress(0.0)
            def progress(frac,msg): bar.progress(min(frac,1.0),text=msg)
            result=run_opt(pack,opts,opts["seed"],progress)
            result["n_days"]=pack["days_used"]; result["n_windows"]=pack["n_windows"]; result["case"]=opts["case"]
            st.session_state["optimization_result"]=result
            bar.progress(1.0,text="자동 튜닝 완료")
        except Exception as exc:
            st.error(f"자동 튜닝 중 오류: {type(exc).__name__}: {exc}")
    result=st.session_state.get("optimization_result")
    pack=st.session_state.get("tune_pack")
    if result:
        st.subheader("자동 튜닝 결과")
        best=result["best"]
        u=result["usage"]
        st.info(f"평가 창 사용: 탐색 단계 학습창 {u['search_used']}/{u['train_total']}개 (후보 {u['search_evals']}개 × 50Hz {u['search_steps']:,}스텝) · "
                f"안정성 검사 학습창 {u['stab_windows']}개 · 최종 검증 검증창 {u['valid_used']}/{u['valid_total']}개 × 노이즈 시드 {u['valid_seeds']}개 · "
                f"일별 파일 {result.get('n_days','-')}/{pack['days_total'] if pack else '-'}일 사용")
        st.caption(f"실험 케이스: {result.get('case','-')} · 목적함수: {result['objective']} · 탐색 시드 {result['seed']} · Kd 로그 샘플링 {'ON' if result['kd_log'] else 'OFF'}")
        if result["stable"] is False:
            st.error("상위 후보 중 안정성 검사(게인 ×1.5/×2, 서보 τ ×0.5/×1.5, 리밋사이클)를 통과한 후보가 없습니다. 아래 값은 점수 1위이며 안정성이 보장되지 않습니다.")
        elif result["stable"]:
            st.success("추천 파라미터는 안정성 검사를 통과한 후보 중 점수가 가장 좋은 값입니다. 아래 값은 시뮬레이터의 현재 센서·서보 가정값에 종속됩니다.")
        mc=st.columns(4)
        mc[0].metric("추천 FF",f"{best['ff']:.3f}")
        mc[1].metric("추천 Kp",f"{best['kp']:.3f}")
        mc[2].metric("추천 Kd",f"{best['kd']:.4f}")
        mc[3].metric("추천 α",f"{best['alpha']:.4f}")
        comp=pd.DataFrame([
            {"검증 구분":name,"검증 점수":m["score"],"자세 RMS비":m["rms_ratio"],
             "Roll RMS":"%.3f°"%m["roll_rms"],"Pitch RMS":"%.3f°"%m["pitch_rms"],
             "Roll 최대각":"%.3f°"%m["roll_max"],"Pitch 최대각":"%.3f°"%m["pitch_max"],
             "서보 포화율":"%.2f%%"%(100*m["saturation"]),"리밋사이클 창 비율":"%.2f"%m["lc_frac"]}
            for name,m in (("현재 파라미터",result["current_valid"]),("추천 파라미터",result["valid"]))])
        st.markdown(f"**학습에 사용하지 않은 검증 구간 결과** (노이즈 시드 {u['valid_seeds']}개 평균)")
        st.dataframe(comp,hide_index=True,width="stretch")
        cur=result["current_valid"]; opt=result["valid"]
        if cur["score"]>0:
            st.metric("검증 점수 개선율",f"{100*(cur['score']-opt['score'])/cur['score']:.1f}%")
        st.markdown("**상위 후보 10개 (탐색 단계 학습창 기준)**")
        st.dataframe(result["top"].round(4),hide_index=True,width="stretch")
        st.button("추천 파라미터를 제어 슬라이더에 적용",on_click=lambda: st.session_state.update({"apply_best_on_next_run":True}),
                  type="primary",width="stretch",key="apply_best_button")
        st.download_button("최적화 후보 및 결과 CSV 저장",result["top"].to_csv(index=False).encode("utf-8-sig"),
                           file_name="akademik_auto_tuning_candidates.csv",mime="text/csv")
        st.caption("서보 τ·최대 각속도·회전한계와 센서 노이즈/바이어스 설정은 최적화 중 고정했습니다. 이 고정값이 실제 하드웨어 사양과 측정치가 아니면 추천값의 신뢰도가 낮아집니다.")

    st.subheader("튜닝 분석")
    t1,t2,t3,t4,t5,t6=st.tabs(["① 수렴·재현성","② 전체 후보·파레토","③ 안정성 검사","④ Kp×Kd 격자·민감도","⑤ 해상상태별 튜닝","⑥ 창 목록·분포"])

    # ---------------------------------------------------------------- ① convergence & reproducibility
    with t1:
        if result:
            h=result["history"]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=h["평가 횟수"],y=h["점수"],mode="markers",name="각 후보 점수",marker=dict(color="#9AA5B1",size=6)))
            fig.add_trace(go.Scatter(x=h["평가 횟수"],y=h["최고 점수"],mode="lines",name="지금까지 최고 점수",line=dict(color=PLATE_COLOR,width=3,shape="hv")))
            fig.update_layout(title="평가 횟수 vs 최고 점수 (수렴 곡선)",height=330,yaxis_type="log",xaxis_title="평가 횟수",yaxis_title="학습 점수(로그)",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="conv_fig")
        else:
            st.info("자동 튜닝을 먼저 실행하면 수렴 곡선이 나타납니다.")
        st.markdown("**시드 반복 실험**: 같은 옵션으로 탐색 시드만 바꿔 최적화를 반복합니다(탐색용 학습창 추출도 시드에 따라 달라집니다).")
        n_rep=st.slider("반복할 시드 수",2,10,5,1,key="seedrep_n")
        if st.button("시드 반복 실험",key="run_seedrep"):
            try:
                pk=get_pack(); runs=[]
                bar=st.progress(0.0)
                for r in range(n_rep):
                    res_=run_opt(pk,opts,opts["seed"]+r,lambda f,m,r=r: bar.progress(min((r+f)/n_rep,1.0),text=f"시드 {opts['seed']+r} · {m}"))
                    runs.append(res_)
                rows=[{"탐색 시드":r_["seed"],"FF":r_["best"]["ff"],"Kp":r_["best"]["kp"],"Kd":r_["best"]["kd"],"α":r_["best"]["alpha"],
                       "학습 점수":r_["train"]["score"],"검증 점수":r_["valid"]["score"],"검증 RMS비":r_["valid"]["rms_ratio"],
                       "안정성 검사":"통과" if r_["stable"] else "실패"} for r_ in runs]
                per=pd.DataFrame(rows)
                stats=pd.DataFrame([{"항목":k,"평균":per[k].mean(),"표준편차":per[k].std(ddof=0),"최솟값":per[k].min(),"최댓값":per[k].max()}
                                    for k in ("FF","Kp","Kd","α","검증 점수")])
                st.session_state["seedrep"]={"per":per,"stats":stats,"hist":[(r_["seed"],r_["history"]) for r_ in runs],"case":opts["case"],"objective":opts["objective"]}
            except Exception as exc:
                st.error(f"시드 반복 실험 오류: {type(exc).__name__}: {exc}")
        sr=st.session_state.get("seedrep")
        if sr:
            st.caption(f"실험 케이스: {sr['case']} · 목적함수: {sr['objective']}")
            st.dataframe(sr["stats"].round(4),hide_index=True,width="stretch")
            st.dataframe(sr["per"].round(4),hide_index=True,width="stretch")
            fig=go.Figure()
            for sd,h in sr["hist"]:
                fig.add_trace(go.Scatter(x=h["평가 횟수"],y=h["최고 점수"],mode="lines",name=f"시드 {sd}",line=dict(shape="hv")))
            fig.update_layout(title="시드별 수렴 곡선",height=320,yaxis_type="log",xaxis_title="평가 횟수",yaxis_title="최고 점수(로그)",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="seedrep_fig")
            st.download_button("시드 반복 결과 CSV 저장",sr["per"].to_csv(index=False).encode("utf-8-sig"),file_name="seed_repeat.csv",mime="text/csv",key="dl_seedrep")

    # ---------------------------------------------------------------- ② all candidates & Pareto
    with t2:
        if result:
            alldf=result["all"]
            x=alldf["서보 각속도 RMS(정규화)"].to_numpy(float); y=alldf["자세 RMS비"].to_numpy(float)
            pm=pareto_mask(x,y)
            fig=go.Figure()
            lc_mask=alldf["리밋사이클 창 비율"].to_numpy(float)>0
            fig.add_trace(go.Scatter(x=x[~lc_mask],y=y[~lc_mask],mode="markers",name="후보",marker=dict(color="#9AA5B1",size=7),
                                     text=[f"FF {a:.2f}, Kp {b:.2f}, Kd {c:.3f}, α {d:.3f}" for a,b,c,d in alldf.loc[~lc_mask,["FF","Kp","Kd","α"]].to_numpy()]))
            fig.add_trace(go.Scatter(x=x[lc_mask],y=y[lc_mask],mode="markers",name="리밋사이클 검출 후보",marker=dict(color=PLATE_OFF_COLOR,size=7,symbol="x")))
            order=np.argsort(x[pm])
            fig.add_trace(go.Scatter(x=x[pm][order],y=y[pm][order],mode="lines+markers",name="파레토 해(비지배해)",line=dict(color=PLATE_COLOR,width=2.5),marker=dict(size=9)))
            rec=alldf["추천"].to_numpy(bool)
            fig.add_trace(go.Scatter(x=x[rec],y=y[rec],mode="markers",name="추천값",marker=dict(color="#F2C14E",size=16,symbol="star",line=dict(color="black",width=1))))
            fig.update_layout(title="서보 각속도 RMS(정규화) vs 자세 RMS비",height=420,xaxis_title="서보 각속도 RMS / 최대 각속도",yaxis_title="자세 RMS비(보정 후/보정 전)",
                              yaxis_type="log",xaxis_type="log",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="pareto_fig")
            st.caption(f"평가한 후보 {len(alldf)}개 중 파레토 해 {int(pm.sum())}개. 왼쪽 아래로 갈수록 적은 서보 움직임으로 수평을 잘 유지합니다.")
            st.dataframe(alldf.round(4),hide_index=True,width="stretch")
            st.download_button("전체 후보 CSV 저장",alldf.to_csv(index=False).encode("utf-8-sig"),file_name="all_candidates.csv",mime="text/csv",key="dl_all")
        else:
            st.info("자동 튜닝을 먼저 실행하면 평가한 모든 후보가 나타납니다.")

    # ---------------------------------------------------------------- ③ stability check
    with t3:
        st.markdown(f"**판정 기준**: 게인 (Kp, Kd) ×{STAB_GAIN_FACTORS[0]:g}·×{STAB_GAIN_FACTORS[1]:g}, 서보 τ ×{STAB_TAU_FACTORS[0]:g}·×{STAB_TAU_FACTORS[1]:g} 각각에서 "
                    f"자세 RMS비(검사 창 전체 합산) < 1.0 이고 포화율 < {100*STAB_SAT_MAX:.0f}%. 기본 조건에서 리밋사이클이 없어야 함 "
                    f"(관절 각속도 부호변화 진동수 > {LC_FLIP_HZ:g} Hz, 또는 2Hz 이상 성분 비율 > {LC_HF_RATIO:g} 이면서 RMS > {LC_HF_RMS:g}°이면 리밋사이클).")
        if result and result.get("stability"):
            st.markdown(f"**추천 파라미터** ({fmt_params(result['best'])})")
            st.dataframe(result["stability"]["table"].round(4),hide_index=True,width="stretch")
        if st.button("현재 슬라이더 값 안정성 검사",key="run_stab_current"):
            try:
                pk=get_pack(); tr,_=split_items(pk)
                st.session_state["stab_current"]={"params":dict(current),"res":stability_check(current,tr[:STAB_WINDOWS],fixed)}
            except Exception as exc:
                st.error(f"안정성 검사 오류: {type(exc).__name__}: {exc}")
        sc_=st.session_state.get("stab_current")
        if sc_:
            st.markdown(f"**현재 슬라이더 값** ({fmt_params(sc_['params'])}) → {'통과' if sc_['res']['passed'] else '실패'}")
            st.dataframe(sc_["res"]["table"].round(4),hide_index=True,width="stretch")

    # ---------------------------------------------------------------- ④ grid & sensitivity
    with t4:
        st.write("FF와 α는 사이드바 슬라이더 값으로 고정하고 Kp×Kd 격자를 평가합니다. 색은 자세 RMS비(검사 창 전체 합산), ✕는 안정성 검사 실패 칸입니다.")
        g1,g2,g3=st.columns(3)
        gn=g1.select_slider("격자 크기",[8,10,12],value=12,key="grid_n")
        kd_axis=g2.radio("Kd 축",["선형","로그"],horizontal=True,key="grid_kdaxis")
        kd_view=g3.select_slider("Kd 최대",[0.06,0.09,0.12,0.2,0.3],value=0.12,key="grid_kdmax")
        if st.button("Kp×Kd 격자 탐색 실행",key="run_grid"):
            try:
                pk=get_pack(); tr,_=split_items(pk)
                sub=tr[:6]
                kps=np.linspace(0.0,4.0,gn)
                kds=np.linspace(0.0,kd_view,gn) if kd_axis=="선형" else np.r_[0.0,np.geomspace(KD_LOG_MIN,kd_view,gn-1)]
                bar=st.progress(0.0)
                gr=grid_scan(ff,alpha,kps,kds,sub,fixed,opts["objective"],lambda f: bar.progress(min(f,1.0),text="격자 평가 중"))
                if result:
                    inp_,sd_,ow_=prepare_inputs(sub)
                    gr["random_best"]=dict(result["best"])
                    gr["random_score"]=evaluate(inp_,sd_,[result["best"]],fixed,opts["objective"],ow_)["summary"][0]["score"]
                gr["objective"]=opts["objective"]
                st.session_state["grid_result"]=gr
            except Exception as exc:
                st.error(f"격자 탐색 오류: {type(exc).__name__}: {exc}")
        gr=st.session_state.get("grid_result")
        if gr:
            ratio=np.minimum(gr["ratio"],1.5)
            xlab=[f"{v:.4f}" for v in gr["kds"]]; ylab=[f"{v:.2f}" for v in gr["kps"]]
            fig=go.Figure(go.Heatmap(z=ratio,x=xlab,y=ylab,colorscale="Viridis_r",zmin=0,zmax=1.5,colorbar=dict(title="RMS비")))
            fi,fj=np.where(~gr["passed"])
            fig.add_trace(go.Scatter(x=[xlab[j] for j in fj],y=[ylab[i] for i in fi],mode="markers",name="안정성 검사 실패",
                                     marker=dict(symbol="x",color="white",size=9,line=dict(color="black",width=1))))
            sc=np.where(gr["passed"],gr["score"],np.inf)
            if not np.isfinite(sc).any(): sc=gr["score"]
            bi,bj=np.unravel_index(int(np.argmin(sc)),sc.shape)
            fig.add_trace(go.Scatter(x=[xlab[bj]],y=[ylab[bi]],mode="markers",name="격자 최적값(안정 통과 중)",marker=dict(symbol="diamond",color=PLATE_COLOR,size=14,line=dict(color="black",width=1))))
            txt=f"격자 최적값: Kp {gr['kps'][bi]:.2f}, Kd {gr['kds'][bj]:.4f}, 점수 {gr['score'][bi,bj]:.4f}"
            if "random_best" in gr:
                rb=gr["random_best"]
                jx=int(np.argmin(np.abs(gr["kds"]-rb["kd"]))); iy=int(np.argmin(np.abs(gr["kps"]-rb["kp"])))
                fig.add_trace(go.Scatter(x=[xlab[jx]],y=[ylab[iy]],mode="markers",name="무작위 탐색 최적값(가장 가까운 칸)",marker=dict(symbol="star",color="#F2C14E",size=16,line=dict(color="black",width=1))))
                diff=100*(gr["random_score"]-gr["score"][bi,bj])/max(abs(gr["score"][bi,bj]),1e-12)
                txt+=f" · 무작위 탐색 최적값({fmt_params(rb)}) 점수 {gr['random_score']:.4f} → 격자 최적 대비 {diff:+.1f}%"
                txt+=" (무작위 탐색값은 FF·α도 튜닝했을 수 있어 음수가 나올 수 있습니다)"
            fig.update_layout(title=f"Kp×Kd 안정 지도 (FF {gr['ff']:.2f}, α {gr['alpha']:.3f} 고정, 창 {gr['n_windows']}개)",height=480,
                              xaxis_title="Kd (s)",yaxis_title="Kp",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="grid_fig")
            st.caption(txt+f" · 목적함수: {gr['objective']}")
            grid_df=pd.DataFrame(gr["ratio"],index=[f"Kp={v}" for v in ylab],columns=[f"Kd={v}" for v in xlab])
            st.download_button("격자 결과 CSV 저장",grid_df.to_csv().encode("utf-8-sig"),file_name="kp_kd_grid.csv",mime="text/csv",key="dl_grid")
        st.markdown("**1D 민감도**: 나머지 파라미터를 자동 튜닝 최적값(없으면 현재 슬라이더 값)에 고정하고 한 파라미터만 바꿉니다. Kd는 슬라이더 최대 0.30까지 봅니다.")
        if st.button("1D 민감도 곡선 계산",key="run_sens"):
            try:
                pk=get_pack(); tr,_=split_items(pk)
                base=dict(result["best"]) if result else dict(current)
                bar=st.progress(0.0)
                st.session_state["sens_result"]={"base":base,"df":sensitivity_curves(base,tr[:6],fixed,opts["objective"],progress_cb=lambda f: bar.progress(min(f,1.0),text="민감도 계산 중"))}
            except Exception as exc:
                st.error(f"민감도 계산 오류: {type(exc).__name__}: {exc}")
        sres=st.session_state.get("sens_result")
        if sres:
            sd=sres["df"]
            fig=make_subplots(rows=1,cols=4,subplot_titles=["FF","Kp","Kd","α"],shared_yaxes=True)
            for i,k in enumerate(["ff","kp","kd","alpha"],1):
                part=sd[sd["파라미터"]==k]
                fig.add_trace(go.Scatter(x=part["값"],y=part["자세 RMS비"],mode="lines+markers",name="자세 RMS비",showlegend=i==1,line=dict(color=PLATE_COLOR)),row=1,col=i)
                lcp=part[part["리밋사이클 창 비율"]>0]
                fig.add_trace(go.Scatter(x=lcp["값"],y=lcp["자세 RMS비"],mode="markers",name="리밋사이클",showlegend=i==1,marker=dict(color=PLATE_OFF_COLOR,symbol="x",size=9)),row=1,col=i)
                fig.add_vline(x=sres["base"][k],line_dash="dot",line_color="gray",row=1,col=i)
            fig.update_yaxes(type="log",title_text="자세 RMS비",row=1,col=1)
            fig.update_layout(height=340,margin=dict(l=10,r=10,t=40,b=10),title=f"1D 민감도 (기준값 {fmt_params(sres['base'])})")
            st.plotly_chart(fig,width="stretch",key="sens_fig")
            st.download_button("민감도 CSV 저장",sd.to_csv(index=False).encode("utf-8-sig"),file_name="sensitivity.csv",mime="text/csv",key="dl_sens")

    # ---------------------------------------------------------------- ⑤ sea-state tuning
    with t5:
        st.caption(HS_NOTE)
        crit=st.radio("조건 구간 기준",["A. WMO 해상상태 코드별","B. Tp 구간별","C. Hs(WMO) × Tp 2차원","D. 하부 IMU 직전 기울기 RMS 구간"],key="bin_crit")
        b1,b2,b3=st.columns(3)
        e1=b1.slider("Tp 경계 1(s)",3.0,20.0,TP_DEFAULT_EDGES[0],0.5,key="tp_e1")
        e2=b2.slider("Tp 경계 2(s)",3.0,20.0,TP_DEFAULT_EDGES[1],0.5,key="tp_e2")
        e3=b3.slider("Tp 경계 3(s)",3.0,20.0,TP_DEFAULT_EDGES[2],0.5,key="tp_e3")
        tp_edges=sorted({e1,e2,e3})
        st.caption("Tp 구간: " + " / ".join(tp_label(b,tp_edges) for b in range(len(tp_edges)+1)))
        d1,d2=st.columns(2)
        prior_min=d1.slider("기준 D: 직전 이동 RMS 시간(분)",2,5,3,1,key="prior_min")
        prior_txt=d2.text_input("기준 D: 기울기 RMS 경계(°, 쉼표 구분)",PRIOR_DEFAULT_EDGES,key="prior_edges")
        m1,m2,m3,m4=st.columns(4)
        min_tr=int(m1.number_input("구간 최소 학습창",1,50,6,1,key="bin_min_tr"))
        min_va=int(m2.number_input("구간 최소 검증창",1,20,2,1,key="bin_min_va"))
        bin_rep=int(m3.slider("오차막대용 시드 반복",1,5,3,1,key="bin_rep"))
        bin_trials=int(m4.slider("구간별 탐색 후보 수",16,60,24,4,key="bin_trials"))
        st.caption("기준 D는 장치가 Hs/Tp를 직접 측정하지 못하므로, 하부 IMU가 측정할 수 있는 직전 기울기 RMS(√(roll²+pitch²), INS 기록 기준)로 구간을 나눕니다. "
                   "직전 자료가 부족한 창(구간 시작부 창)은 제외됩니다. 파랑 자료가 없는 창은 A~C 구간 분석에서 제외되지만 전체 성능 평가에는 포함됩니다.")
        if pack:
            try:
                prior_edges=parse_edges(prior_txt)
                keys,info=assign_bins(pack["items"],crit,tp_edges,prior_min,prior_edges)
                gid,ginfo,log=merge_bins(keys,info,pack["items"],min_tr,min_va)
                prev=pd.DataFrame([{"구간":g["label"],"학습창":g["n_train"],"검증창":g["n_valid"],"분석 가능":"예" if g["enough"] else "아니오(최소 개수 미달)"} for g in ginfo])
                if crit.startswith("A"):
                    present={k[1] for k in info}
                    missing=[{"구간":wmo_label(c),"학습창":0,"검증창":0,"분석 가능":"자료 없음"} for c,_,_,_ in WMO_CODES if c not in present]
                    prev=pd.concat([prev,pd.DataFrame(missing)],ignore_index=True)
                excluded=sum(1 for k in keys if k is None)
                st.markdown(f"**구간 미리보기** (제외된 창 {excluded}개)")
                st.dataframe(prev,hide_index=True,width="stretch")
                for line in log: st.caption("병합: "+line)
            except Exception as exc:
                st.error(f"구간 설정 오류: {exc}")
                gid=None
        else:
            st.info("자동 튜닝 또는 ⑥ 탭의 '튜닝 창 준비'를 먼저 실행하면 구간 미리보기가 나타납니다.")
            gid=None
        if st.button("해상상태별 튜닝 실행",key="run_bins"):
            try:
                pk=get_pack()
                prior_edges=parse_edges(prior_txt)
                keys,info=assign_bins(pk["items"],crit,tp_edges,prior_min,prior_edges)
                gid,ginfo,log=merge_bins(keys,info,pk["items"],min_tr,min_va)
                bar=st.progress(0.0)
                br=run_bin_analysis(pk["items"],gid,ginfo,current,fixed,opts["objective"],opts["case_fixed"],opts["kd_log"],
                                    bin_trials,bin_rep,opts["seed"],lambda f,m: bar.progress(min(f,1.0),text=m))
                br["criterion"]=crit; br["merge_log"]=log; br["case"]=opts["case"]
                st.session_state["bin_result"]=br
            except Exception as exc:
                st.error(f"해상상태별 튜닝 오류: {type(exc).__name__}: {exc}")
        br=st.session_state.get("bin_result")
        if br:
            st.markdown(f"**구간별 최적 게인** · 기준 {br['criterion']} · 케이스 {br['case']} · 목적함수 {br['objective']}")
            for line in br["merge_log"]: st.caption("병합: "+line)
            st.dataframe(br["table"].round(4),hide_index=True,width="stretch")
            for line in br["summary"]: st.info(line)
            st.plotly_chart(heatmap(br["J"],br["labels"],br["gain_names"],"전이 행렬 J(g_i, 구간 j): 행 게인을 열 구간 검증창에 적용한 점수","점수",text_fmt=".3f"),width="stretch",key="bin_J")
            st.plotly_chart(heatmap(100*br["regret"],br["labels"],br["gain_names"],"regret(i→j) = J(g_i, j) / J(g_j*, j) − 1 (%)","regret(%)",zmin=0,text_fmt=".1f",colorscale="Reds"),width="stretch",key="bin_regret")
            reg_df=pd.DataFrame(100*br["regret"],index=br["gain_names"],columns=br["labels"])
            st.dataframe(reg_df.round(2),width="stretch")
            xsel=st.radio("게인–해상상태 그래프 x축",["Hs 평균(m)","Tp 평균(s)"],horizontal=True,key="bin_xaxis")
            tb=br["table"]
            fig=make_subplots(rows=1,cols=3,subplot_titles=["Kp","Kd","FF"])
            for i,(k,e) in enumerate((("Kp","Kp 시드 표준편차"),("Kd","Kd 시드 표준편차"),("FF","FF 시드 표준편차")),1):
                fig.add_trace(go.Scatter(x=tb[xsel],y=tb[k],error_y=dict(type="data",array=tb[e]),mode="markers+text",text=tb["구간"].str.slice(0,10),
                                         textposition="top center",marker=dict(size=10,color=PLATE_COLOR),showlegend=False),row=1,col=i)
            fig.update_layout(height=360,margin=dict(l=10,r=10,t=40,b=10),title="구간 대표값 vs 최적 게인 (오차막대: 시드 반복 표준편차)")
            st.plotly_chart(fig,width="stretch",key="bin_gain_fig")
            st.markdown("**공통 게인 vs 조건별 게인** (각 칸: 검증 RMS비 · 포화율 · regret)")
            st.dataframe(br["comparison"].round(3),hide_index=True,width="stretch")
            st.caption("공통 게인: " + " / ".join(f"{n}: {fmt_params(p)}" for n,p in br["common"].items()))
            c_a,c_b=st.columns(2)
            c_a.download_button("구간별 게인 CSV",br["table"].to_csv(index=False).encode("utf-8-sig"),file_name="gains_by_sea_state.csv",mime="text/csv",key="dl_bins")
            c_b.download_button("공통 vs 조건별 비교 CSV",br["comparison"].to_csv(index=False).encode("utf-8-sig"),file_name="common_vs_binned.csv",mime="text/csv",key="dl_bincomp")
            st.caption(HS_NOTE+" ACE 자료는 특정 해역·기간의 기록이라 일부 WMO 코드는 자료가 없습니다.")

    # ---------------------------------------------------------------- ⑥ windows & distribution
    with t6:
        if st.button("튜닝 창 준비",key="prep_pack"):
            try: get_pack()
            except Exception as exc: st.error(f"튜닝 창 준비 오류: {type(exc).__name__}: {exc}")
        pack=st.session_state.get("tune_pack")
        if pack:
            wt=window_table(pack["items"])
            st.markdown(f"**튜닝 창 목록**: 일별 파일 {pack['days_used']}/{pack['days_total']}일 사용 (균등 추출 {pack['days_chosen']}일) · 창 {pack['n_windows']}개 · 파랑 자료 없음 {int(wt['WMO 코드'].isna().sum())}개")
            if pack["days_used"]<pack["days_total"]:
                st.caption("창 수 상한(180개) 때문에 일부 날짜만 쓰지만, 날짜를 전체 기간에서 균등 간격으로 골라 특정 기간이 빠지지 않습니다.")
            st.dataframe(wt.round(3),hide_index=True,width="stretch")
            st.download_button("창 목록 CSV",wt.to_csv(index=False).encode("utf-8-sig"),file_name="tuning_windows.csv",mime="text/csv",key="dl_windows")
            cats=[str(c) for c,_,_,_ in WMO_CODES]+["자료 없음"]
            wt["WMO 라벨"]=wt["WMO 코드"].apply(lambda v: "자료 없음" if pd.isna(v) else str(int(v)))
            fig=go.Figure()
            for part,color in (("학습",SHIP_COLOR),("검증",PLATE_COLOR)):
                cnt=wt[wt["구분"]==part]["WMO 라벨"].value_counts()
                fig.add_trace(go.Bar(x=cats,y=[int(cnt.get(c,0)) for c in cats],name=part,marker_color=color))
            fig.update_layout(barmode="group",title="WMO 코드별 창 개수",height=320,xaxis_title="WMO 해상상태 코드",yaxis_title="창 개수",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="wmo_bar")
            ok=wt.dropna(subset=["Hs(m)"])
            fig=go.Figure(go.Scatter(x=ok["Tp(s)"],y=ok["Hs(m)"],mode="markers",marker=dict(color=ok["WMO 코드"],colorscale="Turbo",cmin=0,cmax=9,size=8,
                                     colorbar=dict(title="WMO")),text=ok["날짜"]+" "+ok["시작 시각(UTC)"]))
            fig.update_layout(title="Hs–Tp 분포 (색 = WMO 코드)",height=360,xaxis_title="Tp (s)",yaxis_title="Hs (m)",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="hs_tp_fig")
            st.caption(HS_NOTE)
        else:
            st.info("자동 튜닝을 실행하거나 '튜닝 창 준비'를 누르면 창 목록이 나타납니다.")

    with st.expander("실시간 재생과 자동 튜닝 평가의 일치 검증 (consistency_check)", expanded=False):
        st.write("현재 재생 중인 항해 구간에서 튜닝 창과 같은 방식으로 창을 뽑아, 같은 파라미터·같은 노이즈 시드로 "
                 "① 자동 튜닝 평가 경로(simulate_batch)와 ② 실시간 재생 경로(actual_motion_at + control_step)를 각각 돌립니다. "
                 f"RMS는 live_performance와 같은 식으로 계산하며 차이가 {100*CONSISTENCY_TOL:.0f}% 이내면 통과입니다.")
        cc1,cc2=st.columns(2)
        n_check=cc1.slider("검증 창 수",1,3,2,1,key="cc_windows")
        cc_seed=int(cc2.number_input("노이즈 시드",0,99999,NOISE_SEED_BASE,1,key="cc_seed"))
        if st.button("일치 검증 실행",key="run_consistency"):
            seg_now=st.session_state.get("current_actual_df")
            wins=consistency_windows(seg_now,n_check) if seg_now is not None else []
            if not wins:
                st.warning("현재 구간이 너무 짧아(60개 기록 미만) 검증 창을 만들 수 없습니다.")
            else:
                param_sets=[("현재 슬라이더 값",dict(current))]
                if result: param_sets.append(("자동 튜닝 추천값",dict(result["best"])))
                tables=[]
                with st.spinner("두 경로를 각각 50Hz로 계산하는 중… (실시간 경로는 창당 수 초 걸립니다)"):
                    for wi,w in enumerate(wins):
                        for name,pars in param_sets:
                            t_=consistency_check(w,pars,fixed,cc_seed)
                            t_.insert(0,"파라미터",name); t_.insert(0,"창",f"{wi+1} ({len(w)}개 기록, {float(w['elapsed_s'].iloc[-1]-w['elapsed_s'].iloc[0]):.0f}s)")
                            tables.append(t_)
                st.session_state["consistency_result"]=pd.concat(tables,ignore_index=True)
        cres=st.session_state.get("consistency_result")
        if cres is not None:
            st.dataframe(cres.round(4),hide_index=True,width="stretch")
            worst=float(cres["차이(%)"].max())
            if (cres["판정(≤5%)"]=="통과").all():
                st.success(f"모든 창·파라미터에서 통과했습니다. 최대 차이 {worst:.4f}%")
            else:
                st.error(f"허용 차이 5%를 넘는 항목이 있습니다. 최대 차이 {worst:.2f}%")
            st.download_button("일치 검증 결과 CSV 저장",cres.to_csv(index=False).encode("utf-8-sig"),
                               file_name="consistency_check.csv",mime="text/csv",key="dl_consistency")

else:
    # ---------------------------------------------------------------- 5-6 virtual comparison
    st.divider()
    with st.expander("가상 파도 대조 실험: 해상상태별 최적 게인 (Hs × Tp)", expanded=False):
        st.warning("가상 파도는 파도 경사 기반 단순화 모델이며 실선 RAO 모델이 아닙니다. 이 결과는 실제 항해 데이터 결과와 따로 해석해야 합니다.")
        st.write(f"Hs {', '.join(f'{h:g}' for h in VIRTUAL_HS)} m(WMO 3~6 대표값) × Tp {', '.join(f'{t:g}' for t in VIRTUAL_TP)} s의 16개 조건에서 "
                 "가상 창을 만들고 조건마다 자동 튜닝을 실행합니다. 이어서 각 조건의 게인을 다른 조건에 적용한 regret으로 "
                 "Hs와 Tp 중 어느 변수가 최적 게인에 더 큰 영향을 주는지 비교합니다.")
        vopts=tuning_options("virt",expanded=True)
        v1,v2,v3=st.columns(3)
        v_trials=int(v1.slider("조건별 탐색 후보 수",16,48,20,4,key="virt_trials"))
        v_train=int(v2.slider("조건별 학습창 수",2,6,3,1,key="virt_ntrain"))
        v_dur=float(v3.select_slider("창 길이(s)",[60,120,180,240],value=120,key="virt_dur"))
        st.caption("흔들림 크기는 기존 가상 모드 프리셋 배율(Hs 0.5/2/4 m → 0.6/2.0/3.5)을 Hs에 대해 보간한 값 × 사이드바 '흔들림 크기 배율'입니다. 파향 25°, 성분 5개.")
        if st.button("가상 대조 실험 실행",key="run_virtual_bins"):
            try:
                bar=st.progress(0.0)
                st.session_state["virtual_bins"]=run_virtual_analysis(current,fixed,vopts["objective"],vopts["case_fixed"],vopts["kd_log"],
                                                                      v_trials,v_train,v_dur,gain,vopts["seed"],lambda f,m: bar.progress(min(f,1.0),text=m))
            except Exception as exc:
                st.error(f"가상 대조 실험 오류: {type(exc).__name__}: {exc}")
        vr=st.session_state.get("virtual_bins")
        if vr:
            st.info(vr["summary"])
            st.dataframe(vr["table"].round(4),hide_index=True,width="stretch")
            tb=vr["table"]
            for k in ("Kp","Kd","FF"):
                piv=tb.pivot(index="Hs(m)",columns="Tp(s)",values=k)
                st.plotly_chart(heatmap(piv.to_numpy(),[f"Tp {c:g}s" for c in piv.columns],[f"Hs {i:g}m" for i in piv.index],
                                        f"가상 조건별 최적 {k}",k,text_fmt=".3f",colorscale="Blues"),width="stretch",key=f"virt_{k}")
            st.markdown("**변수별 영향도** (최적 게인의 변화 폭, 탐색 범위 대비 %)")
            st.dataframe(vr["influence"].round(2),hide_index=True,width="stretch")
            labels=[f"Hs{h:g}/Tp{t:g}" for h,t in vr["conds"]]
            st.plotly_chart(heatmap(100*vr["regret"],labels,labels,"가상 조건 전이 regret(%) (행 게인 → 열 조건)","regret(%)",zmin=0,text_fmt=".0f",colorscale="Reds"),width="stretch",key="virt_regret")
            st.download_button("가상 대조 실험 CSV",tb.to_csv(index=False).encode("utf-8-sig"),file_name="virtual_sea_state_gains.csv",mime="text/csv",key="dl_virt")
            st.caption("가상 파도는 실선 RAO 모델이 아님. 실제 항해 데이터의 해상상태별 결과와 섞어서 해석하지 마세요.")

with st.expander("모델·데이터 출처와 해석상의 한계"):
    st.markdown("""
**계산 흐름:** 선박 Roll/Pitch → 가상 IMU → 상보필터 → FF+PD → 서보 모델 → 상판 자세

**자동 데이터 관리**
- 첫 실행 때 Zenodo에서 INS ZIP(약 646.9MB)과 Hs/Tp CSV를 앱 캐시에 한 번 다운로드합니다.
- 원본 ZIP은 압축 해제하지 않고, 날짜별 파일을 ZIP 안에서 차례대로 읽습니다.
- 날짜순으로 연속 구간을 인덱싱하며, 구간 사이 긴 시간 공백을 가로질러 보간하지 않습니다.
- 현재 구간이 끝나면 자동으로 다음 날짜/구간으로 넘어갑니다.
- GitHub에는 대용량 원본 데이터를 올리지 않습니다.

**시뮬레이션 기반 자동 튜닝(무작위 탐색 + 국소 탐색)**
- FF, Kp, Kd, α만 조정합니다. 실험 케이스로 FF=1 고정 또는 FF=0(PD만)도 선택할 수 있습니다.
- τ, 최대 각속도, 기계적 회전한계, 가상 센서 노이즈/바이어스는 사용자가 정한 값으로 고정합니다.
- 일별 파일을 전체 기간에서 균등하게 골라 대표 4분 창을 추출합니다. 학습 구간에서 후보를 탐색하고, 일부 날짜는 검증용으로 분리합니다.
- 자동 튜닝 평가는 실시간 재생과 같은 50Hz 제어 루프, 같은 선형 보간, 같은 노이즈 생성 순서를 씁니다(일치 검증 expander에서 확인 가능).
- 탐색 단계는 학습창 일부(최대 30개)만 쓰고, 최종 검증은 검증창 전체 × 노이즈 시드 3개 이상의 평균을 씁니다.
- 목적함수는 제약 조건 방식(자세 RMS비 최소화, 포화율 ≤1%·리밋사이클 없음 위반 시 페널티)과 가중합 방식 중 고를 수 있습니다.
- 서보 부담 지표는 관절 각속도 RMS, 각가속도 RMS, 2Hz 이상 고주파 명령 비율입니다(관절각 자체는 수평 유지에 필요한 동작이라 벌점에서 뺐습니다).
- 최종 추천값은 게인 ×1.5/×2, 서보 τ ×0.5/×1.5, 리밋사이클 검사를 통과한 후보 중에서 고릅니다.

**중요한 과학적 한계**
- 공개 INS는 ACE 기간 중 수집한 1Hz 항해 기록입니다. 제공기관은 원자료가 추가 보정되지 않았고 품질검사도 수행되지 않았다고 설명합니다.
- 기록 PC 시간이 GPS 및 다른 장비와 동기화되었는지는 확실하지 않다고 README에 기재되어 있습니다.
- Hs/Tp는 선박 Heave 운동의 5분 구간 스펙트럼과 선박별 RAO를 이용해 재구성한 자료입니다. 독립 부표의 동시 직접 측정값이 아닙니다.
- INS 자료에는 실제 제작할 캡스톤 플랫폼의 IMU 원시 자이로/가속도 신호가 들어 있지 않습니다. 센서 오차는 모델 가정값입니다.
- 본 프로그램의 서보 모델은 1차 지연·속도 제한·각도 제한을 단순화해 표현합니다. 실제 DS51150의 응답, 부하, 기어 유격, 전원 강하, 구조 탄성은 실측으로 검증해야 합니다.
- D항은 상판 자이로(관절 각속도 포함)를 그대로 쓰므로 Kd가 커지면 빠른 내부 루프가 생겨 진동합니다. Kd 탐색 상한은 이 안정 경계를 반영했습니다.
- 따라서 자동 최적값은 **이 시뮬레이터 모델에 대한 후보값**입니다. 실물 적용 전에는 낮은 이득부터 단계적으로 시험하고, 독립된 실제 IMU·서보 시험 데이터로 재튜닝해야 합니다.

**출처 및 라이선스**
- Thomas & Pina Estany (2020), ACE INS, Zenodo DOI 10.5281/zenodo.3733228, CC BY 4.0.
- Derkani et al. (2021), ACE reconstructed sea state, Zenodo DOI 10.5281/zenodo.4541564, CC BY 4.0.
- 해상상태 분류: WMO Sea State Code(WMO 코드표 3700, Douglas 척도 기반) — 유의파고 Hs 기준 코드 0~9.
- [ACE INS 전체 원자료](https://zenodo.org/records/3733228) · [Hs/Tp 재구성 자료](https://zenodo.org/records/4541564) · [AARI 선박 정보](https://www.aari.ru/eng)
""")
