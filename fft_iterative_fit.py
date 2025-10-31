#!/usr/bin/env python3
"""
Iterative FFT-basierte SPL-Kalibrierung
======================================

Ziel dieses Skripts ist es, das aus der Simulation abgeleitete Schallspektrum
systematisch an die experimentellen SPL-Daten anzunähern. Der Ablauf orientiert
sich an den Signalaufbereitungsschritten aus Jaeger et al. (2008) sowie Perot
et al. (2013):

1. Welch-Analyse mit Hann-Fenster (nperseg=512, 50 % Überlappung) zur
   Erzeugung eines stabilen Basisspektrums.
2. Globale Pegelanpassung (Least-Squares im linearen Druck), um
   Kalibrierunterschiede zwischen Simulation und Messung auszugleichen.
3. Frequenzabhängige Korrekturen auf Basis geglätteter Residuen, damit vor
   allem im Niederfrequenzbereich (< 200 Hz) die Messdaten besser getroffen
   werden, ohne das Spektrum zu verrauschen.

Das Skript speichert Plot, Kennzahlen und CSV-Auszüge unter ``results/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interaktives Backend für Skriptbetrieb
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

P_REF = 20e-6  # Referenzdruck für SPL in Luft


@dataclass
class IterationResult:
    """Speichert Zwischenstände der iterativen Anpassung."""

    name: str
    spl_sim: np.ndarray
    spl_interp: np.ndarray
    diff_db: np.ndarray
    total_correction_sim_db: np.ndarray
    total_correction_exp_db: np.ndarray
    metrics: Dict[str, Dict[str, float]]
    applied_correction_exp: np.ndarray


def read_simulation_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Liest Zeit- und Druckdaten aus der Simulation.

    Erwartetes Format: zwei Spalten (Zeit [s], Schalldruck [Pa]) mit
    Header-Zeilen, die mit '(' beginnen.
    """
    times: List[float] = []
    pressures: List[float] = []

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
                time_val = float(parts[0])
                pressure_val = float(parts[1])
            except ValueError:
                continue
            times.append(time_val)
            pressures.append(pressure_val)

    if not times:
        raise ValueError(f"Keine numerischen Daten in {path} gefunden.")

    return np.asarray(times), np.asarray(pressures)


def describe_signal(time: np.ndarray, pressure: np.ndarray) -> None:
    """Gibt grundlegende Kennzahlen des Zeitsignals aus."""
    duration = time[-1] - time[0]
    dt = np.diff(time)
    fs = 1.0 / np.mean(dt)
    print("Zeitsignal:")
    print(f"  Samples          : {len(time)}")
    print(f"  Dauer            : {duration:.6f} s")
    print(f"  Mittelwert Druck : {np.mean(pressure):.6e} Pa")
    print(f"  RMS (AC)         : {np.std(pressure, ddof=0):.6e} Pa")
    print(f"  RMS (inkl. DC)   : {math.sqrt(np.mean(pressure**2)):.6e} Pa")
    print(f"  Abtastrate       : {fs:.2f} Hz (Δt ≈ {np.mean(dt)*1e3:.3f} ms)")
    print(f"  Nyquist-Frequenz : {fs/2:.1f} Hz")


def read_experimental_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Lädt experimentelle SPL-Daten (Frequenz [Hz]; SPL [dB])."""
    data = np.loadtxt(path, delimiter=";")
    if data.shape[1] != 2:
        raise ValueError(f"Unerwartetes Datenformat in {path}")
    freq = data[:, 0]
    spl = data[:, 1]
    print("\nExperimentelle Daten:")
    print(f"  Frequenzen       : {freq[0]:.3f} ... {freq[-1]:.3f} Hz")
    print(f"  Punkte           : {len(freq)} (Δf = {freq[1] - freq[0]:.3f} Hz)")
    print(f"  SPL-Spanne       : {spl.min():.2f} ... {spl.max():.2f} dB")
    return freq, spl


def welch_spectrum(
    pressure: np.ndarray,
    fs: float,
    nperseg: int = 512,
    overlap: float = 0.5,
    detrend: str = "constant",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechnet den Amplitudenverlauf mittels Welch-Methode.

    Rückgabewert ist (Frequenz, RMS-Amplitude pro Frequenzbin).
    """
    noverlap = int(nperseg * overlap)
    freq, psd = signal.welch(
        pressure,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",
        return_onesided=True,
    )
    if len(freq) < 2:
        raise ValueError("Welch-Auswertung liefert zu wenige Frequenzpunkte.")
    df = freq[1] - freq[0]
    amplitude_rms = np.sqrt(psd * df)
    return freq, amplitude_rms


def amplitude_to_spl(amplitude: np.ndarray, p_ref: float = P_REF) -> np.ndarray:
    """Wandelt RMS-Amplituden (Pa) in SPL [dB] um."""
    amplitude_safe = np.maximum(amplitude, 1e-20)
    return 20.0 * np.log10(amplitude_safe / p_ref)


def compute_gain_db(
    freq_sim: np.ndarray,
    amp_sim: np.ndarray,
    freq_exp: np.ndarray,
    spl_exp: np.ndarray,
    freq_range: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Bestimmt die globale Verstärkung, die die simulierten Amplituden optimal
    (Least-Squares im linearen Druck) an die Messwerte angleicht.
    """
    f_min, f_max = freq_range
    amp_exp = P_REF * 10 ** (spl_exp / 20.0)
    amp_sim_interp = np.interp(
        freq_exp, freq_sim, amp_sim, left=amp_sim[0], right=amp_sim[-1]
    )
    mask = (freq_exp >= f_min) & (freq_exp <= f_max)
    if not np.any(mask):
        return 0.0, 1.0
    a = amp_sim_interp[mask]
    b = amp_exp[mask]
    denominator = np.dot(a, a)
    if denominator <= 0:
        return 0.0, 1.0
    scale = float(np.dot(a, b) / denominator)
    if scale <= 0:
        return 0.0, 1.0
    gain_db = 20.0 * math.log10(scale)
    return gain_db, scale


def smooth_residual(
    freq_exp: np.ndarray,
    residual_db: np.ndarray,
    window_hz: float,
    freq_floor: float,
    freq_max: float,
    clip_db: Optional[float],
) -> np.ndarray:
    """
    Glättet die Differenz (Experiment - Simulation) über die Frequenz.

    - window_hz bestimmt die effektive Glättungsbreite (Gaussian, ~FWHM).
    - Unterhalb freq_floor wird der Korrekturwert konstant gehalten, weil
      die Zeitsignal-Länge dort keine verlässliche Information bietet.
    - clip_db (optional) begrenzt Extremkorrekturen, um unrealistische Peaks zu
      vermeiden.
    """
    if len(freq_exp) < 3:
        return residual_db.copy()

    freq_step = np.mean(np.diff(freq_exp))
    sigma = max(window_hz / (2.355 * freq_step), 1.0)
    smoothed = gaussian_filter1d(residual_db, sigma=sigma, mode="nearest")
    smoothed = smoothed.copy()

    if freq_max is not None:
        mask = freq_exp > freq_max
        if np.any(mask):
            if np.any(~mask):
                smoothed[mask] = smoothed[np.where(~mask)[0][-1]]
            else:
                smoothed[mask] = 0.0

    if freq_floor is not None and freq_floor > 0.0:
        idx = np.searchsorted(freq_exp, freq_floor)
        if idx < len(smoothed):
            smoothed[:idx] = smoothed[idx]

    if clip_db is not None:
        smoothed = np.clip(smoothed, -abs(clip_db), abs(clip_db))

    return smoothed


def compute_metrics(
    freq: np.ndarray,
    diff_db: np.ndarray,
    freq_range: Tuple[float, float],
    bands: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, float]]:
    """Berechnet RMSE/MAE/Mittelwert usw. für verschiedene Frequenzbereiche."""
    metrics: Dict[str, Dict[str, float]] = {}
    f_min, f_max = freq_range
    mask_total = (freq >= f_min) & (freq <= f_max)
    if np.any(mask_total):
        subset = diff_db[mask_total]
        metrics["overall"] = {
            "rmse": float(np.sqrt(np.mean(subset**2))),
            "mae": float(np.mean(np.abs(subset))),
            "mean": float(np.mean(subset)),
            "max": float(np.max(subset)),
            "min": float(np.min(subset)),
        }

    for label, (band_min, band_max) in bands.items():
        mask = (freq >= band_min) & (freq <= band_max)
        if not np.any(mask):
            continue
        segment = diff_db[mask]
        metrics[label] = {
            "rmse": float(np.sqrt(np.mean(segment**2))),
            "mae": float(np.mean(np.abs(segment))),
            "mean": float(np.mean(segment)),
            "max": float(np.max(segment)),
            "min": float(np.min(segment)),
        }

    return metrics


def iterative_fit(
    freq_sim: np.ndarray,
    amp_sim: np.ndarray,
    freq_exp: np.ndarray,
    spl_exp: np.ndarray,
    smoothing_windows: List[float],
    freq_range_gain: Tuple[float, float],
    freq_range_metrics: Tuple[float, float],
    bands: Dict[str, Tuple[float, float]],
    freq_floor: float,
    freq_max: float,
    clip_db: float,
) -> Tuple[List[IterationResult], np.ndarray, float]:
    """
    Führt die iterative Anpassung durch und liefert die Zwischenstände sowie
    die kumulierte Korrektur (in dB) auf dem Simulationsfrequenzraster.
    """
    iterations: List[IterationResult] = []
    total_corr_sim = np.zeros_like(freq_sim)
    total_corr_exp = np.zeros_like(freq_exp)

    # Baseline (keine Korrektur)
    spl_base_sim = amplitude_to_spl(amp_sim)
    spl_base_exp = np.interp(freq_exp, freq_sim, spl_base_sim)
    diff_db = spl_base_exp - spl_exp
    baseline_metrics = compute_metrics(freq_exp, diff_db, freq_range_metrics, bands)
    iterations.append(
        IterationResult(
            name="Baseline (Welch)",
            spl_sim=spl_base_sim.copy(),
            spl_interp=spl_base_exp.copy(),
            diff_db=diff_db.copy(),
            total_correction_sim_db=total_corr_sim.copy(),
            total_correction_exp_db=total_corr_exp.copy(),
            metrics=baseline_metrics,
            applied_correction_exp=np.zeros_like(freq_exp),
        )
    )

    # Globale Verstärkung
    gain_db, scale = compute_gain_db(freq_sim, amp_sim, freq_exp, spl_exp, freq_range_gain)
    total_corr_exp += gain_db
    total_corr_sim += gain_db
    spl_gain_sim = spl_base_sim + total_corr_sim
    spl_gain_exp = spl_base_exp + total_corr_exp
    diff_gain = spl_gain_exp - spl_exp
    gain_metrics = compute_metrics(freq_exp, diff_gain, freq_range_metrics, bands)
    iterations.append(
        IterationResult(
            name="Nach globalem Gain",
            spl_sim=spl_gain_sim.copy(),
            spl_interp=spl_gain_exp.copy(),
            diff_db=diff_gain.copy(),
            total_correction_sim_db=total_corr_sim.copy(),
            total_correction_exp_db=total_corr_exp.copy(),
            metrics=gain_metrics,
            applied_correction_exp=np.full_like(freq_exp, gain_db),
        )
    )

    current_spl_exp = spl_gain_exp
    print(f"\nGlobale Kalibrierung: Gain = {gain_db:+.2f} dB (Skalierung {scale:.3f})")

    # Frequenzabhängige Korrekturen
    for window_hz in smoothing_windows:
        residual = spl_exp - current_spl_exp
        correction_exp = smooth_residual(
            freq_exp,
            residual,
            window_hz=window_hz,
            freq_floor=freq_floor,
            freq_max=freq_max,
            clip_db=clip_db,
        )
        total_corr_exp += correction_exp
        total_corr_sim = np.interp(
            freq_sim,
            freq_exp,
            total_corr_exp,
            left=total_corr_exp[0],
            right=total_corr_exp[-1],
        )
        current_spl_sim = spl_base_sim + total_corr_sim
        current_spl_exp = spl_base_exp + total_corr_exp
        current_diff = current_spl_exp - spl_exp
        current_metrics = compute_metrics(freq_exp, current_diff, freq_range_metrics, bands)

        iterations.append(
            IterationResult(
                name=f"+ Glättung {int(round(window_hz))} Hz",
                spl_sim=current_spl_sim.copy(),
                spl_interp=current_spl_exp.copy(),
                diff_db=current_diff.copy(),
                total_correction_sim_db=total_corr_sim.copy(),
                total_correction_exp_db=total_corr_exp.copy(),
                metrics=current_metrics,
                applied_correction_exp=correction_exp.copy(),
            )
        )

        lf_stats = current_metrics.get("20-100 Hz")
        lf_info = (
            f", LF-RMSE={lf_stats['rmse']:.2f} dB"
            if lf_stats is not None
            else ""
        )
        print(
            f"Korrektur {int(round(window_hz))} Hz: "
            f"RMSE (0-1 kHz) = {current_metrics['overall']['rmse']:.2f} dB{lf_info}"
        )

    return iterations, total_corr_sim, gain_db


def plot_iterations(
    freq_exp: np.ndarray,
    spl_exp: np.ndarray,
    freq_sim: np.ndarray,
    iterations: List[IterationResult],
    gain_db: float,
    output_path: Path,
    freq_xlim: Tuple[float, float],
) -> None:
    """Visualisiert Messung, Simulation und Differenzen vor/nach der Korrektur."""
    if not iterations:
        raise ValueError("Keine Iterationsdaten zum Plotten vorhanden.")

    baseline = iterations[0]
    final = iterations[-1]
    gain_step = iterations[1] if len(iterations) > 1 else None

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # SPL-Verläufe
    ax = axes[0]
    ax.plot(freq_exp, spl_exp, label="Experiment (microA)", color="C3", linewidth=1.5)
    ax.plot(
        freq_sim,
        baseline.spl_sim,
        label="Simulation (Baseline)",
        color="0.5",
        linewidth=1.3,
        alpha=0.9,
    )
    ax.plot(
        freq_sim,
        final.spl_sim,
        label="Simulation (korrigiert)",
        color="C0",
        linewidth=1.8,
    )
    ax.set_ylabel("SPL [dB re 20 µPa]")
    ax.set_xlim(*freq_xlim)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Vergleich SPL: Experiment vs. Simulation")

    # Differenzen
    ax = axes[1]
    ax.plot(
        freq_exp,
        baseline.diff_db,
        label="Baseline - Experiment",
        color="0.5",
        linewidth=1.3,
    )
    if gain_step is not None:
        ax.plot(
            freq_exp,
            gain_step.diff_db,
            label="Nach globalem Gain",
            color="C1",
            linewidth=1.0,
            alpha=0.9,
        )
    ax.plot(
        freq_exp,
        final.diff_db,
        label="Korrigiert - Experiment",
        color="C0",
        linewidth=1.5,
    )
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Δ SPL [dB]")
    ax.set_xlim(*freq_xlim)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Abweichung Simulation - Experiment")

    # Korrekturverlauf
    ax = axes[2]
    ax.plot(
        freq_exp,
        np.full_like(freq_exp, gain_db),
        label=f"Globales Gain ({gain_db:+.2f} dB)",
        color="C1",
        linewidth=1.0,
        linestyle="--",
    )
    for result in iterations[2:]:
        ax.plot(
            freq_exp,
            result.applied_correction_exp,
            label=result.name,
            linewidth=1.2,
        )
    total_correction_exp = final.total_correction_exp_db
    ax.plot(
        freq_exp,
        total_correction_exp,
        label="Summe der Korrekturen",
        color="C0",
        linewidth=2.0,
        alpha=0.8,
    )
    ax.set_xlabel("Frequenz [Hz]")
    ax.set_ylabel("Angewandte Korrektur [dB]")
    ax.set_xlim(*freq_xlim)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Korrekturen im Frequenzbereich")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot gespeichert unter: {output_path}")


def save_csv_output(
    freq_exp: np.ndarray,
    spl_exp: np.ndarray,
    iterations: List[IterationResult],
    freq_sim: np.ndarray,
    output_path: Path,
) -> None:
    """Schreibt Mess- und Simulationsergebnisse in eine CSV-Datei."""
    baseline_interp = iterations[0].spl_interp
    final_interp = iterations[-1].spl_interp
    total_correction_exp = iterations[-1].total_correction_exp_db
    diff_final = final_interp - spl_exp

    header = (
        "Frequency_Hz,SPL_Experiment_dB,SPL_Simulation_Baseline_dB,"
        "SPL_Simulation_Final_dB,Diff_Final_minus_Exp_dB,Applied_Correction_dB"
    )
    data = np.column_stack(
        [
            freq_exp,
            spl_exp,
            baseline_interp,
            final_interp,
            diff_final,
            total_correction_exp,
        ]
    )
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")
    print(f"Numerische Auswertung gespeichert unter: {output_path}")


def save_metrics(iterations: List[IterationResult], output_path: Path) -> None:
    """Persistiert die Kennzahlen der einzelnen Iterationsschritte als Text."""
    with output_path.open("w", encoding="utf-8") as handle:
        for result in iterations:
            handle.write(f"{result.name}\n")
            for label, stats in result.metrics.items():
                handle.write(
                    f"  {label:>12}: "
                    f"RMSE={stats['rmse']:6.2f} dB, "
                    f"MAE={stats['mae']:6.2f} dB, "
                    f"Mean={stats['mean']:6.2f} dB, "
                    f"Min={stats['min']:6.2f} dB, "
                    f"Max={stats['max']:6.2f} dB\n"
                )
            handle.write("\n")
    print(f"Kennzahlen gespeichert unter: {output_path}")


def main() -> None:
    base_path = Path(__file__).resolve().parent
    data_dir = base_path / "data"
    results_dir = base_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sim_path = data_dir / "phi10 theta17 A.txt"
    exp_path = data_dir / "microA.csv"

    time, pressure = read_simulation_data(sim_path)
    describe_signal(time, pressure)

    freq_exp, spl_exp = read_experimental_data(exp_path)

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    pressure_ac = pressure - np.mean(pressure)

    freq_sim, amp_sim = welch_spectrum(
        pressure_ac, fs=fs, nperseg=512, overlap=0.5, detrend="constant"
    )

    smoothing_windows = [60.0, 25.0]  # Hz
    freq_range_gain = (20.0, 1000.0)
    freq_range_metrics = (0.0, 1000.0)
    bands = {
        "0-20 Hz": (0.0, 20.0),
        "20-100 Hz": (20.0, 100.0),
        "100-500 Hz": (100.0, 500.0),
        "500-1000 Hz": (500.0, 1000.0),
    }

    iterations, total_correction_db, gain_db = iterative_fit(
        freq_sim=freq_sim,
        amp_sim=amp_sim,
        freq_exp=freq_exp,
        spl_exp=spl_exp,
        smoothing_windows=smoothing_windows,
        freq_range_gain=freq_range_gain,
        freq_range_metrics=freq_range_metrics,
        bands=bands,
        freq_floor=0.0,
        freq_max=1200.0,
        clip_db=None,
    )

    final_metrics = iterations[-1].metrics["overall"]
    print(
        "\nEndergebnis (0-1000 Hz): "
        f"RMSE={final_metrics['rmse']:.2f} dB, "
        f"MAE={final_metrics['mae']:.2f} dB, "
        f"Mean={final_metrics['mean']:.2f} dB"
    )
    if "20-100 Hz" in iterations[-1].metrics:
        lf = iterations[-1].metrics["20-100 Hz"]
        print(
            f"  Niederfrequenz (20-100 Hz): RMSE={lf['rmse']:.2f} dB, "
            f"MAE={lf['mae']:.2f} dB, Mean={lf['mean']:.2f} dB"
        )
    print(
        "Hinweis: Frequenzen < 2 Hz bleiben aufgrund der kurzen Simulationsdauer "
        "unsicher und werden lediglich mit einer geglätteten Korrektur versehen."
    )

    plot_path = results_dir / "fft_iterative_fit.png"
    plot_iterations(
        freq_exp=freq_exp,
        spl_exp=spl_exp,
        freq_sim=freq_sim,
        iterations=iterations,
        gain_db=gain_db,
        output_path=plot_path,
        freq_xlim=(0.0, 1200.0),
    )

    csv_path = results_dir / "fft_iterative_fit.csv"
    save_csv_output(
        freq_exp=freq_exp,
        spl_exp=spl_exp,
        iterations=iterations,
        freq_sim=freq_sim,
        output_path=csv_path,
    )

    metrics_path = results_dir / "fft_iterative_fit_metrics.txt"
    save_metrics(iterations, metrics_path)


if __name__ == "__main__":
    main()
