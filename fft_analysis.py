#!/usr/bin/env python3
"""
FFT-Analyse und SPL-Berechnung für HVAC-Duct Schalldruckdaten
Basierend auf Methodologie aus Jaeger et al. 2008 und Perot et al. 2013
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

def read_simulation_data(filename):
    """
    Liest die simulierten Schalldruckdaten ein
    Format: Zeit [s], Schalldruck [Pa]
    """
    print(f"Lese Simulationsdaten aus {filename}...")

    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Überspringe Header-Zeilen
            if line.startswith('(') or 'title' in line or 'labels' in line or line.strip() == '':
                continue
            # Versuche Daten zu parsen
            try:
                parts = line.strip().split()
                if len(parts) >= 2:
                    time = float(parts[0])
                    pressure = float(parts[1])
                    data.append([time, pressure])
            except ValueError:
                continue

    data = np.array(data)
    print(f"  Anzahl Datenpunkte: {len(data)}")

    if len(data) > 0:
        print(f"  Zeitbereich: {data[0, 0]:.4f} - {data[-1, 0]:.4f} s")
        print(f"  Druckbereich: {data[:, 1].min():.6f} - {data[:, 1].max():.6f} Pa")

        # Berechne Sampling-Rate
        dt = np.mean(np.diff(data[:, 0]))
        fs = 1.0 / dt
        print(f"  Mittlere Abtastrate: {fs:.1f} Hz")
        print(f"  Zeitschritt: {dt*1000:.3f} ms")

    return data[:, 0], data[:, 1]

def read_experimental_data(filename):
    """
    Liest die experimentellen SPL-Daten ein
    Format: Frequenz [Hz]; SPL [dB]
    """
    print(f"\nLese experimentelle Daten aus {filename}...")

    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(';')
                if len(parts) == 2:
                    freq_hz = float(parts[0].replace(',', '.'))
                    spl = float(parts[1].replace(',', '.'))
                    data.append([freq_hz, spl])  # Frequency already in Hz

    data = np.array(data)
    print(f"  Anzahl Frequenzpunkte: {len(data)}")
    if len(data) > 0:
        print(f"  Frequenzbereich: {data[0, 0]:.1f} - {data[-1, 0]:.1f} Hz")
        print(f"  SPL-Bereich: {data[:, 1].min():.1f} - {data[:, 1].max():.1f} dB")
        print(f"  Frequenzauflösung: {data[1, 0] - data[0, 0]:.3f} Hz")

    return data[:, 0], data[:, 1]

def perform_fft_analysis(time, pressure, window_type='hann', nperseg=512, overlap_ratio=0.5):
    """
    Führt FFT-Analyse mit Windowing und Overlap durch
    Basiert auf Parametern aus Jaeger et al. 2008
    """
    print(f"\nFFT-Analyse mit {window_type} Window, nperseg={nperseg}, overlap={overlap_ratio*100}%...")

    # Berechne Sampling-Rate
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    # Entferne DC-Offset
    pressure_ac = pressure - np.mean(pressure)

    # Welch's Method für Power Spectral Density
    # Dies entspricht der Methode mit überlappenden Fenstern aus dem Paper
    noverlap = int(nperseg * overlap_ratio)

    frequencies, psd = signal.welch(pressure_ac, fs=fs, window=window_type,
                                    nperseg=nperseg, noverlap=noverlap,
                                    detrend='linear', scaling='density')

    # Konvertiere PSD zu Amplitude für SPL-Berechnung
    # PSD hat Einheit Pa²/Hz, wir brauchen Pa_rms
    amplitude = np.sqrt(psd * (frequencies[1] - frequencies[0]))  # RMS amplitude per frequency bin

    print(f"  Frequenzauflösung: {frequencies[1] - frequencies[0]:.2f} Hz")
    print(f"  Max. Frequenz: {frequencies[-1]:.1f} Hz")

    return frequencies, amplitude, psd

def calculate_spl(amplitude, p_ref=20e-6):
    """
    Berechnet Sound Pressure Level (SPL) in dB
    SPL = 20 * log10(p_rms / p_ref)
    p_ref = 20 µPa (Standard-Referenzdruck für Luft)
    """
    # Vermeidung von log(0) durch Hinzufügen eines kleinen Wertes
    amplitude_safe = np.maximum(amplitude, 1e-20)
    spl = 20 * np.log10(amplitude_safe / p_ref)
    return spl

def plot_results(freq_sim, spl_sim, freq_exp, spl_exp):
    """
    Erstellt Vergleichsplots zwischen Simulation und Experiment
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: SPL Vergleich (linear scale)
    ax1 = axes[0, 0]
    ax1.plot(freq_exp, spl_exp, 'ro-', label='Experiment', markersize=4, alpha=0.7)
    ax1.plot(freq_sim, spl_sim, 'b-', label='FFT Simulation', linewidth=2)
    ax1.set_xlabel('Frequenz [Hz]')
    ax1.set_ylabel('SPL [dB]')
    ax1.set_title('Sound Pressure Level - Vergleich')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1000])

    # Plot 2: SPL Vergleich (log scale)
    ax2 = axes[0, 1]
    ax2.semilogx(freq_exp, spl_exp, 'ro-', label='Experiment', markersize=4, alpha=0.7)
    ax2.semilogx(freq_sim, spl_sim, 'b-', label='FFT Simulation', linewidth=2)
    ax2.set_xlabel('Frequenz [Hz]')
    ax2.set_ylabel('SPL [dB]')
    ax2.set_title('Sound Pressure Level - Logarithmische Frequenzachse')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim([10, 10000])

    # Plot 3: Differenz
    ax3 = axes[1, 0]
    # Interpoliere Simulationsdaten auf experimentelle Frequenzpunkte
    spl_sim_interp = np.interp(freq_exp, freq_sim, spl_sim)
    diff = spl_sim_interp - spl_exp
    ax3.plot(freq_exp, diff, 'g.-', markersize=4)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Frequenz [Hz]')
    ax3.set_ylabel('Differenz SPL [dB]')
    ax3.set_title('Differenz (Simulation - Experiment)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1000])

    # Plot 4: Narrow-band Spektrum
    ax4 = axes[1, 1]
    ax4.plot(freq_sim, spl_sim, 'b-', linewidth=1, alpha=0.7)
    ax4.set_xlabel('Frequenz [Hz]')
    ax4.set_ylabel('SPL [dB]')
    ax4.set_title('Narrow-band Spektrum (FFT)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 2000])
    ax4.set_ylim([0, max(spl_sim[freq_sim < 2000]) + 10])

    plt.tight_layout()
    return fig

def analyze_peaks(frequencies, spl, threshold_db=3):
    """
    Identifiziert dominante Frequenzen im Spektrum
    """
    print("\nPeak-Analyse:")

    # Finde Peaks
    peaks, properties = signal.find_peaks(spl, height=np.mean(spl), prominence=threshold_db)

    if len(peaks) > 0:
        # Sortiere nach SPL
        sorted_indices = np.argsort(spl[peaks])[::-1]

        print(f"  Top 10 dominante Frequenzen:")
        for i, idx in enumerate(sorted_indices[:10]):
            peak_idx = peaks[idx]
            print(f"    {i+1}. {frequencies[peak_idx]:.1f} Hz: {spl[peak_idx]:.1f} dB")
    else:
        print("  Keine signifikanten Peaks gefunden")

def analyze_fft_resolution(time, pressure):
    """
    Analysiert verschiedene FFT-Parameter und deren Einfluss auf die Auflösung
    """
    print("\n" + "=" * 60)
    print("ANALYSE DER FFT-AUFLÖSUNG")
    print("=" * 60)

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    total_time = time[-1] - time[0]
    n_samples = len(time)

    print(f"\nZeitsignal-Eigenschaften:")
    print(f"  Gesamtdauer: {total_time:.3f} s")
    print(f"  Anzahl Samples: {n_samples}")
    print(f"  Abtastrate: {fs:.1f} Hz")
    print(f"  Nyquist-Frequenz: {fs/2:.1f} Hz")

    # Verschiedene FFT-Konfigurationen
    configs = [
        (256, 0.5),   # Kleineres Fenster
        (512, 0.5),   # Original (Jaeger)
        (1024, 0.5),  # Größeres Fenster
        (2048, 0.5),  # Noch größeres Fenster
        (4096, 0.75), # Sehr großes Fenster mit mehr Überlappung
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    pressure_ac = pressure - np.mean(pressure)

    for idx, (nperseg, overlap) in enumerate(configs):
        if nperseg > n_samples:
            continue

        noverlap = int(nperseg * overlap)
        freq, psd = signal.welch(pressure_ac, fs=fs, window='hann',
                                 nperseg=nperseg, noverlap=noverlap,
                                 detrend='linear', scaling='density')

        amplitude = np.sqrt(psd * (freq[1] - freq[0]))
        spl = calculate_spl(amplitude)

        # Berechne theoretische Auflösung
        freq_resolution = fs / nperseg
        time_per_segment = nperseg / fs * 1000  # in ms
        n_segments = (n_samples - noverlap) / (nperseg - noverlap)

        ax = axes[idx]
        ax.plot(freq[freq < 500], spl[freq < 500], linewidth=1.5)
        ax.set_xlabel('Frequenz [Hz]')
        ax.set_ylabel('SPL [dB]')
        ax.set_title(f'nperseg={nperseg}, overlap={overlap*100:.0f}%\n'
                    f'Δf={freq_resolution:.1f} Hz, T_seg={time_per_segment:.1f} ms')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 500])

        print(f"\nKonfiguration {idx+1}: nperseg={nperseg}, overlap={overlap*100:.0f}%")
        print(f"  Frequenzauflösung: {freq_resolution:.2f} Hz")
        print(f"  Zeit pro Segment: {time_per_segment:.1f} ms")
        print(f"  Anzahl Segmente (mit Überlappung): ~{n_segments:.0f}")
        print(f"  Niedrigste auflösbare Frequenz: ~{freq_resolution:.1f} Hz")

    # Lösche unbenutzte Subplots
    for idx in range(len(configs), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Einfluss der FFT-Parameter auf die Spektralauflösung', fontsize=14)
    plt.tight_layout()

    return fig

def main():
    """
    Hauptfunktion für die FFT-Analyse
    """
    print("=" * 60)
    print("FFT-ANALYSE FÜR HVAC-DUCT SCHALLDRUCKDATEN")
    print("Basierend auf Jaeger et al. 2008 und Perot et al. 2013")
    print("=" * 60)

    # Dateipfade
    sim_file = 'data/phi10 theta17 A.txt'
    exp_file = 'data/microA.csv'

    # 1. Lade Daten
    time, pressure = read_simulation_data(sim_file)
    freq_exp, spl_exp = read_experimental_data(exp_file)

    # Analysiere verschiedene FFT-Auflösungen
    resolution_fig = analyze_fft_resolution(time, pressure)
    resolution_fig.savefig('fft_resolution_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nAuflösungsanalyse gespeichert in: fft_resolution_analysis.png")

    # 2. FFT-Analyse mit Parametern aus Jaeger et al. 2008
    # Paper verwendet: DFT-length: 512, Window: Hanning, Overlap: 50%
    freq_sim, amplitude, psd = perform_fft_analysis(time, pressure,
                                                    window_type='hann',
                                                    nperseg=512,
                                                    overlap_ratio=0.5)

    # 3. Berechne SPL
    spl_sim = calculate_spl(amplitude)

    # 4. Analysiere Peaks
    analyze_peaks(freq_sim, spl_sim)

    # 5. Statistik
    print("\nStatistik:")
    # Interpoliere für Vergleich
    spl_sim_interp = np.interp(freq_exp, freq_sim, spl_sim)
    rmse = np.sqrt(np.mean((spl_sim_interp - spl_exp)**2))
    mean_diff = np.mean(spl_sim_interp - spl_exp)

    print(f"  RMSE: {rmse:.2f} dB")
    print(f"  Mittlere Differenz: {mean_diff:.2f} dB")
    print(f"  Max. Differenz: {np.max(np.abs(spl_sim_interp - spl_exp)):.2f} dB")

    # 6. Visualisierung
    fig = plot_results(freq_sim, spl_sim, freq_exp, spl_exp)

    # Speichere Ergebnisse
    output_file = 'fft_results.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nErgebnisse gespeichert in: {output_file}")

    # Speichere numerische Ergebnisse
    results_df = pd.DataFrame({
        'Frequency_Hz': freq_sim,
        'SPL_dB': spl_sim,
        'PSD_Pa2_Hz': psd
    })
    results_df.to_csv('fft_results.csv', index=False)
    print(f"Numerische Ergebnisse gespeichert in: fft_results.csv")
    print("\nAnalyse abgeschlossen!")

if __name__ == "__main__":
    main()