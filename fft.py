##------ Import ------##
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig

##------ Variablen ------##
Data = 'phi10 theta17 A.txt'        # Datenquelle
L_window = 1024                     # Fensterlänge
window = 'hann'                     # Fensterart: Hann-Fenster
overlap = 0.5                       # Überöappung %/100%



##------ Daten einlesen ------##

t_list, pa_list = [], []
with open(Data) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('('):
            continue
        parts = line.split()
        if len(parts) != 2:
            #Debug-Ausgabe zur Problemzeile:
            print(f"Übersprungene Zeile: {line}")
            continue
        a, b = parts
        t_list.append(float(a))
        pa_list.append(float(b))

t_physisch = np.array(t_list)   # pysikalische Zeit in Spaltenvektor schreiben in s
pa_sim = np.array(pa_list)      # akustischer Druck in Spaltenvektor schreiben in Pa
#print(t_physisch.shape, pa_sim.shape) #Test eigelesene Daten

signal = pa_sim.copy()
#print(signal)  #Signal Test 


##------ Datenverarbeitung ------##

# Abtastfrequnz bestimmen
delta_t_diff = np.diff(t_physisch)      # Berechnung der Zeitschritte in s
delta_t_mean = np.mean(delta_t_diff)    # Berechnung des gemittelten Zeitschritts in s
Fs = 1.0/delta_t_mean                     # Berechnung Abtastfrequenz in Hz
#print(Fs)
L = signal.size                         # Berechnung Signallänge / Signaleinträge
#print(L)


##------ FFT ------##
L_overlap = int(L_window*overlap)        # Länger der Fensterüberlappung

# Referenzwert für akustischen Druck (Standard in Luft)
p_ref = 20e-6  # 20 µPa

# mit Short-Time Fourier Transform (STFT)
f, t, Zxx = spsig.stft(signal,
                        fs=Fs,
                        window=window,
                        nperseg=L_window,
                        noverlap=L_overlap,
                        boundary=None,      # kein Zero-Padding an Rändern
                        padded=False)       # kein Padding vor/nach dem Signal

# Mittelung des Spektrums über alle Zeitsegmente
mean_spectrum = np.mean(np.abs(Zxx), axis=1)  # Mittelwert über Zeiten, pro Frequenz

# Darstellung des mittleren Spektrums in dB (SPL)
mean_spectrum_dB = 20 * np.log10(mean_spectrum / p_ref + 1e-12)


#mit Welch-Algorithmus

# Welch PSD berechnen
f, Pxx = spsig.welch(signal,
                     fs=Fs,
                     window=window,
                     nperseg=L_window,
                     noverlap=L_overlap,
                     detrend='constant',
                     scaling='density',   # PSD in Einheit Ein²/Hz
                     return_onesided=True)

# Referenzwert für PSD (Pa²/Hz)
P_ref = (20e-6)**2  # (20 µPa)² = 4e-10 Pa²/Hz

# In dB umrechnen (10*log10 für Leistung/PSD)
Pxx_dB = 10 * np.log10(Pxx / P_ref + 1e-20)

##------ Plot ------##


plt.figure(figsize=(8,4))
plt.plot(t_physisch, signal)
plt.xlabel('Zeit s')
plt.ylabel('akustischer Druck in Pa')
plt.title('Signal aus CFD mit FWH')
plt.xlim(t_physisch[0], t_physisch[-1])
plt.grid(True, which='both', ls='--' )
plt.tight_layout()
plt.savefig("plot1.png")
#plt.show()

plt.figure(figsize=(8,4))
plt.semilogx(f, mean_spectrum_dB)
plt.xlabel('Frequenz in Hz')
plt.ylabel('SPL in dB re 20 µPa')
plt.title('Frequenzspektrum mit STFT\n(Hann, 50% Überlappung)')
#plt.xlim(0, Fs/2)
plt.grid(True, which='both', ls='--' )
plt.tight_layout()
plt.savefig("plot2.png")
#plt.show()


plt.figure(figsize=(8,4))
plt.semilogx(f, Pxx_dB)       # oder plt.plot(f, Pxx_dB)
plt.xlabel('Frequenz in Hz')
plt.ylabel('PSD in dB re (20 µPa)²/Hz')
plt.title('Power Spectral Density (PSD) mit Welch \n(Hann, 50% Überlappung)')
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig("plot3.png")
#plt.show()