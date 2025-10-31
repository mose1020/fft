#!/usr/bin/env python3
"""
FFT-Analyse mit festen Parametern aus der Literatur:
- DFT-Length: 512
- Window: Hann
- Overlap: 50 %
- Ziel-Frequenzauflösung: ~4 Hz

Da die simulierten Daten mit ~25 kHz abgetastet wurden, wird das Zeitsignal
vor der Welch-Analyse per resample_poly auf ~2.048 kHz heruntergesampelt.
Dadurch entspricht 512 FFT-Punkte einer 4-Hz-Binbreite, wie im Paper angegeben.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

P_REF = 20e-6


def load_simulation(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times = []
    pressures = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("(")
                or stripped.startswith(")")
                or "title" in stripped.lower()
                or "labels" in stripped.lower()
            ):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                times.append(float(parts[0]))
                pressures.append(float(parts[1]))
            except ValueError:
                continue
    if not times:
        raise ValueError(f"Keine Messwerte in {path}")
    return np.asarray(times), np.asarray(pressures)


def load_experiment(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=";")
    if data.shape[1] != 2:
        raise ValueError(f"Unerwartetes Format in {path}")
    return data[:, 0], data[:, 1]


def resample_to_target_fs(time: np.ndarray, pressure: np.ndarray, target_fs: float):
    dt = np.diff(time)
    fs = 1.0 / np.mean(dt)
    ratio = Fraction(target_fs / fs).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    resampled = signal.resample_poly(pressure, up, down, window=("kaiser", 5.0))
    fs_new = fs * up / down
    t0 = time[0]
    resampled_time = t0 + np.arange(len(resampled)) / fs_new
    print(f"Original fs: {fs:.2f} Hz, Ziel fs: {target_fs:.2f} Hz, effektiv: {fs_new:.2f} Hz")
    print(f"Resampling-Faktoren: up={up}, down={down}, Samples neu: {len(resampled)}")
    return resampled_time, resampled, fs_new


def welch_fft(pressure: np.ndarray, fs: float):
    pressure_ac = pressure - np.mean(pressure)
    freq, psd = signal.welch(
        pressure_ac,
        fs=fs,
        window="hann",
        nperseg=512,
        noverlap=256,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    df = freq[1] - freq[0]
    amplitude = np.sqrt(psd * df)
    spl = 20.0 * np.log10(np.maximum(amplitude, 1e-20) / P_REF)
    return freq, spl, amplitude


def compare_to_experiment(
    freq_sim: np.ndarray,
    spl_sim: np.ndarray,
    amp_sim: np.ndarray,
    freq_exp: np.ndarray,
    spl_exp: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    spl_interp = np.interp(freq_exp, freq_sim, spl_sim)
    diff = spl_interp - spl_exp
    mask = (freq_exp >= 0.0) & (freq_exp <= 1000.0)
    rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))
    mae = float(np.mean(np.abs(diff[mask])))
    mean = float(np.mean(diff[mask]))
    lf_mask = (freq_exp >= 20.0) & (freq_exp <= 100.0)
    lf_rmse = float(np.sqrt(np.mean(diff[lf_mask] ** 2)))
    print(
        f"Vergleich 0-1 kHz: RMSE={rmse:.2f} dB, MAE={mae:.2f} dB, Mittelwert={mean:+.2f} dB"
    )
    print(f"  Niederfrequenz (20-100 Hz): RMSE={lf_rmse:.2f} dB")
    base_metrics = (rmse, mae, lf_rmse)

    amp_sim_interp = np.interp(freq_exp, freq_sim, amp_sim)
    amp_exp = P_REF * 10 ** (spl_exp / 20.0)
    mask_cal = (freq_exp >= 20.0) & (freq_exp <= 1000.0)
    numerator = float(np.dot(amp_sim_interp[mask_cal], amp_exp[mask_cal]))
    denominator = float(np.dot(amp_sim_interp[mask_cal], amp_sim_interp[mask_cal]))
    scale = numerator / denominator if denominator > 0 else 1.0
    gain_db = 20.0 * math.log10(scale) if scale > 0 else 0.0
    spl_scaled = spl_sim + gain_db
    spl_scaled_interp = spl_interp + gain_db
    diff_scaled = spl_scaled_interp - spl_exp
    rmse_s = float(np.sqrt(np.mean(diff_scaled[mask] ** 2)))
    mae_s = float(np.mean(np.abs(diff_scaled[mask])))
    lf_rmse_s = float(np.sqrt(np.mean(diff_scaled[lf_mask] ** 2)))
    print(
        f"Kalibriert (Gain {gain_db:+.2f} dB): RMSE={rmse_s:.2f} dB, MAE={mae_s:.2f} dB"
    )
    print(f"  LF-RMSE={lf_rmse_s:.2f} dB")
    scaled_metrics = (rmse_s, mae_s, lf_rmse_s)
    return base_metrics, scaled_metrics, gain_db


def plot_comparison(
    freq_sim: np.ndarray,
    spl_sim: np.ndarray,
    spl_scaled: np.ndarray,
    freq_exp: np.ndarray,
    spl_exp: np.ndarray,
    output_path: Path,
    rmse_pair: Tuple[float, float],
    lf_pair: Tuple[float, float],
    gain_db: float,
) -> None:
    spl_interp = np.interp(freq_exp, freq_sim, spl_sim)
    spl_scaled_interp = np.interp(freq_exp, freq_sim, spl_scaled)
    diff = spl_interp - spl_exp
    diff_scaled = spl_scaled_interp - spl_exp
    rmse_base, rmse_scaled = rmse_pair
    lf_base, lf_scaled = lf_pair

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(freq_exp, spl_exp, label="Experiment (microA)", color="C3", linewidth=1.5)
    ax.plot(freq_sim, spl_sim, label="Simulation (ohne Kalibrierung)", color="C0", linewidth=1.7)
    ax.plot(freq_sim, spl_scaled, label=f"Simulation (+{gain_db:.2f} dB Gain)", color="C1", linewidth=1.5, linestyle="--")
    ax.set_ylabel("SPL [dB re 20 µPa]")
    ax.set_xlim(0, 1000)
    ax.set_ylim(min(np.min(spl_exp), np.min(spl_sim)) - 5, max(np.max(spl_exp), np.max(spl_sim)) + 5)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("SPL-Vergleich: FFT mit festen Parametern vs. Experiment")

    ax = axes[1]
    ax.plot(freq_exp, diff, color="C2", linewidth=1.3, label=f"Ohne Kalibrierung (RMSE {rmse_base:.2f} dB, LF {lf_base:.2f} dB)")
    ax.plot(freq_exp, diff_scaled, color="C1", linewidth=1.3, linestyle="--", label=f"Mit Gain (RMSE {rmse_scaled:.2f} dB, LF {lf_scaled:.2f} dB)")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Frequenz [Hz]")
    ax.set_ylabel("Δ SPL [dB]")
    ax.set_xlim(0, 1000)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Differenz (Simulation - Experiment)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot gespeichert unter: {output_path}")


def main() -> None:
    base_path = Path(__file__).resolve().parent
    data_dir = base_path / "data"
    results_dir = base_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sim_file = data_dir / "phi10 theta17 A.txt"
    exp_file = data_dir / "microA.csv"

    time, pressure = load_simulation(sim_file)
    freq_exp, spl_exp = load_experiment(exp_file)

    target_fs = 4.0 * 512.0
    time_rs, pressure_rs, fs_rs = resample_to_target_fs(time, pressure, target_fs)

    freq_sim, spl_sim, amplitude = welch_fft(pressure_rs, fs_rs)

    (rmse, mae, lf_rmse), (rmse_s, mae_s, lf_rmse_s), gain_db = compare_to_experiment(
        freq_sim, spl_sim, amplitude, freq_exp, spl_exp
    )

    spl_scaled = spl_sim + gain_db

    csv_data = np.column_stack([freq_sim, spl_sim, amplitude])
    csv_path = results_dir / "fft_fixed_params_4hz.csv"
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header="Frequency_Hz,SPL_dB,Amplitude_RMS_Pa",
        comments="",
    )
    print(f"Spectrum exportiert: {csv_path}")

    plot_path = results_dir / "fft_fixed_params_4hz.png"
    plot_comparison(
        freq_sim,
        spl_sim,
        spl_scaled,
        freq_exp,
        spl_exp,
        plot_path,
        rmse_pair=(rmse, rmse_s),
        lf_pair=(lf_rmse, lf_rmse_s),
        gain_db=gain_db,
    )


if __name__ == "__main__":
    main()
