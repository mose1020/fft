#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def main() -> None:
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Simulationsdaten einlesen
    time = []
    pressure = []
    with (data_dir / "phi10 theta17 A.txt").open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("(") or s.startswith(")") or "title" in s.lower() or "labels" in s.lower():
                continue
            parts = s.split()
            if len(parts) >= 2:
                time.append(float(parts[0]))
                pressure.append(float(parts[1]))
    time = np.asarray(time)
    pressure = np.asarray(pressure)

    # Messdaten einlesen
    freq_exp, spl_exp = np.loadtxt(data_dir / "microA.csv", delimiter=";", unpack=True)

    # FFT mit 4 Hz Auflösung vorbereiten: jedes 12. Sample verwenden
    dt = np.diff(time)
    fs_orig = 1.0 / np.mean(dt)
    decimation = 12
    pressure_rs = pressure[::decimation]
    time_rs = time[::decimation]
    fs_rs = fs_orig / decimation
    print(
        f"Original fs: {fs_orig:.2f} Hz  ->  decimiert fs: {fs_rs:.2f} Hz (jeder {decimation}. Wert)"
    )
    print(f"Effektive Frequenzauflösung: {fs_rs/512:.3f} Hz pro Bin")

    # Welch-Analyse
    freq_sim, psd = signal.welch(
        pressure_rs - np.mean(pressure_rs),
        fs=fs_rs,
        window="hann",
        nperseg=512,
        noverlap=256,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    df = freq_sim[1] - freq_sim[0]
    amp_sim = np.sqrt(psd * df)
    spl_sim = 20.0 * np.log10(np.maximum(amp_sim, 1e-20) / 20e-6)

    # Globales Gain für besseren Vergleich
    amp_sim_interp = np.interp(freq_exp, freq_sim, amp_sim)
    amp_exp = 20e-6 * 10 ** (spl_exp / 20.0)
    mask = (freq_exp >= 20.0) & (freq_exp <= 1000.0)
    gain = np.dot(amp_sim_interp[mask], amp_exp[mask]) / np.dot(
        amp_sim_interp[mask], amp_sim_interp[mask]
    )
    gain_db = 20.0 * np.log10(gain) if gain > 0 else 0.0
    spl_sim_cal = spl_sim + gain_db
    print(f"Globales Gain: {gain_db:+.2f} dB")

    # Kennzahlen
    spl_interp = np.interp(freq_exp, freq_sim, spl_sim_cal)
    diff = spl_interp - spl_exp
    mask_total = (freq_exp >= 0.0) & (freq_exp <= 1000.0)
    rmse = np.sqrt(np.mean(diff[mask_total] ** 2))
    lf_mask = (freq_exp >= 20.0) & (freq_exp <= 100.0)
    lf_rmse = np.sqrt(np.mean(diff[lf_mask] ** 2))
    print(f"RMSE (0-1 kHz): {rmse:.2f} dB  |  RMSE (20-100 Hz): {lf_rmse:.2f} dB")

    # Frequenz-Plot
    fig_freq, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(freq_exp, spl_exp, label="Experiment", color="C3")
    axes[0].plot(freq_sim, spl_sim, label="Simulation roh", color="0.6")
    axes[0].plot(
        freq_sim,
        spl_sim_cal,
        label=f"Simulation + Gain ({gain_db:+.2f} dB)",
        color="C0",
    )
    axes[0].set_ylabel("SPL [dB]")
    axes[0].set_xlim(0, 1000)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(freq_exp, diff, color="C0")
    axes[1].axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Frequenz [Hz]")
    axes[1].set_ylabel("Δ SPL [dB]")
    axes[1].set_xlim(0, 1000)
    axes[1].grid(alpha=0.3)

    plot_freq_path = results_dir / "fft_fixed_params_4hz_simple.png"
    fig_freq.tight_layout()
    fig_freq.savefig(plot_freq_path, dpi=160)
    plt.close(fig_freq)
    print(f"SPL-Plot gespeichert: {plot_freq_path}")

    # CSV ausgeben
    csv_path = results_dir / "fft_fixed_params_4hz_simple.csv"
    np.savetxt(
        csv_path,
        np.column_stack([freq_sim, spl_sim_cal]),
        delimiter=",",
        header="Frequency_Hz,SPL_dB",
        comments="",
    )
    print(f"CSV gespeichert: {csv_path}")

    # Zeitbereichsvergleich plotten
    fig_time, axes_time = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes_time[0].plot(time, pressure, color="C3", linewidth=0.8)
    axes_time[0].set_title("Originalsignal (Zeitsignal)")
    axes_time[0].set_ylabel("Druck [Pa]")
    axes_time[0].grid(alpha=0.3)

    axes_time[1].plot(time_rs, pressure_rs, color="C0", linewidth=0.8)
    axes_time[1].set_title("Decimiertes Signal (jeder 12. Wert)")
    axes_time[1].set_xlabel("Zeit [s]")
    axes_time[1].set_ylabel("Druck [Pa]")
    axes_time[1].grid(alpha=0.3)

    fig_time.tight_layout()
    time_plot_path = results_dir / "fft_fixed_params_4hz_simple_time.png"
    fig_time.savefig(time_plot_path, dpi=160)
    plt.close(fig_time)
    print(f"Zeitbereichsplot gespeichert: {time_plot_path}")


if __name__ == "__main__":
    main()
