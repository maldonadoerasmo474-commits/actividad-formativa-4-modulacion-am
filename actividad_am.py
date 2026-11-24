#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulación de modulación en amplitud (AM) y análisis de perturbaciones.

Requisitos:
    pip install numpy scipy matplotlib

Ejecución:
    python actividad_am.py
    python actividad_am.py --output-dir resultados

El programa crea gráficas PNG, un CSV de métricas y un resumen JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert


# ----------------------------- Parámetros -----------------------------
FS = 200_000                 # Frecuencia de muestreo (muestras/s)
DURATION = 0.04              # Duración de la simulación (s)
FM = 500                     # Frecuencia de la señal de información (Hz)
FC = 10_000                  # Frecuencia de la portadora (Hz)
AC = 1.0                     # Amplitud de la portadora (V, normalizada)
MOD_INDEX = 0.70             # Índice de modulación (70 %)
SNR_DB = 15.0                # Relación señal/ruido solicitada
ATTENUATION = 0.50           # Ganancia de canal: 50 % de amplitud
CLIP_LEVEL = 1.25            # Límite del recortador (unidades normalizadas)
RANDOM_SEED = 2026           # Semilla para repetir exactamente el ruido


def single_sided_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Calcula espectro de amplitud unilateral, sin ventana.

    La duración contiene un número entero de periodos de fm y fc, por lo que
    el espectro de la señal ideal presenta fuga espectral mínima.
    """
    n = len(x)
    spectrum = np.fft.rfft(x)
    amplitude = np.abs(spectrum) / n
    if n > 1:
        amplitude[1:-1] *= 2
        if n % 2:
            amplitude[-1] *= 2
    frequency = np.fft.rfftfreq(n, d=1 / fs)
    return frequency, amplitude


def demodulate_envelope(received: np.ndarray, channel_gain: float = 1.0) -> np.ndarray:
    """Demodulación ideal por envolvente usando la transformada de Hilbert.

    Se supone conocida la ganancia del canal y se elimina el pequeño offset
    residual para comparar la forma de onda recuperada con el mensaje original.
    """
    analytic = hilbert(received)
    envelope = np.abs(analytic)
    estimate = (envelope / (channel_gain * AC) - 1.0) / MOD_INDEX
    estimate = estimate - np.mean(estimate)
    return estimate


def compute_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Calcula RMSE, correlación y SNR de error de la señal recuperada."""
    error = reference - estimate
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    signal_power = float(np.mean(reference ** 2))
    error_power = float(np.mean(error ** 2))
    snr_recovered = float(10 * np.log10(signal_power / error_power)) if error_power > 0 else float("inf")
    corr = float(np.corrcoef(reference, estimate)[0, 1])
    return {
        "mse": mse,
        "rmse": rmse,
        "snr_error_db": snr_recovered,
        "correlacion": corr,
    }


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulación AM con ruido, distorsión y atenuación.")
    parser.add_argument("--output-dir", default="outputs", help="Carpeta para guardar resultados.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generación de la señal de información y la portadora.
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    message = np.cos(2 * np.pi * FM * t)  # Mensaje normalizado entre -1 y 1.
    carrier = AC * np.cos(2 * np.pi * FC * t)

    # 2. AM convencional: s(t) = Ac [1 + mu*m(t)] cos(2*pi*fc*t).
    am = AC * (1 + MOD_INDEX * message) * np.cos(2 * np.pi * FC * t)

    # 3. Ruido blanco gaussiano aditivo (AWGN) con SNR controlada.
    rng = np.random.default_rng(RANDOM_SEED)
    signal_power = np.mean(am ** 2)
    noise_power = signal_power / (10 ** (SNR_DB / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=n)
    am_noisy = am + noise

    # 4. Distorsión no lineal: recorte simétrico de los picos de la AM.
    am_clipped = np.clip(am, -CLIP_LEVEL, CLIP_LEVEL)
    clipping_rate = float(np.mean(np.abs(am) > CLIP_LEVEL) * 100)

    # 5. Atenuación lineal del canal (la ganancia reduce la amplitud a la mitad).
    am_attenuated = ATTENUATION * am
    power_reduction_db = float(10 * np.log10(ATTENUATION ** 2))

    # 6. Demodulación por envolvente con calibración ideal de la ganancia.
    recovered_clean = demodulate_envelope(am, 1.0)
    recovered_noisy = demodulate_envelope(am_noisy, 1.0)
    recovered_clipped = demodulate_envelope(am_clipped, 1.0)
    recovered_attenuated = demodulate_envelope(am_attenuated, ATTENUATION)

    recovered = {
        "Señal ideal": recovered_clean,
        "Con ruido (15 dB)": recovered_noisy,
        "Con recorte": recovered_clipped,
        "Con atenuación (50 %)": recovered_attenuated,
    }
    metrics = {name: compute_metrics(message, estimate) for name, estimate in recovered.items()}

    # 7. Gráfica de mensaje, portadora y señal AM (primeros 4 ms).
    zoom = t <= 0.004
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t[zoom] * 1000, message[zoom], linewidth=1.6)
    style_axes(axes[0], "Señal de información m(t)", "Tiempo (ms)", "Amplitud normalizada")
    axes[1].plot(t[zoom] * 1000, carrier[zoom], linewidth=0.9)
    style_axes(axes[1], "Portadora c(t) de 10 kHz", "Tiempo (ms)", "Amplitud (V normalizada)")
    axes[2].plot(t[zoom] * 1000, am[zoom], linewidth=0.9, label="Señal AM")
    envelope_upper = AC * (1 + MOD_INDEX * message[zoom])
    envelope_lower = -envelope_upper
    axes[2].plot(t[zoom] * 1000, envelope_upper, "--", linewidth=1.2, label="Envolvente")
    axes[2].plot(t[zoom] * 1000, envelope_lower, "--", linewidth=1.2)
    style_axes(axes[2], "Señal AM con portadora y su envolvente", "Tiempo (ms)", "Amplitud (V normalizada)")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "01_senal_y_modulacion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 8. Espectro de frecuencia AM.
    freq, amp_spec = single_sided_spectrum(am, FS)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(freq / 1000, amp_spec, linewidth=1.2)
    ax.set_xlim(0, 15)
    for f in (FC - FM, FC, FC + FM):
        ax.axvline(f / 1000, linestyle="--", alpha=0.75)
    ax.annotate("Banda lateral inferior\n9.5 kHz", xy=((FC - FM) / 1000, 0.35),
                xytext=(-105, 48), textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.annotate("Portadora\n10.0 kHz", xy=(FC / 1000, 1.0),
                xytext=(18, -5), textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.annotate("Banda lateral superior\n10.5 kHz", xy=((FC + FM) / 1000, 0.35),
                xytext=(35, 48), textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    style_axes(ax, "Espectro de amplitud de la señal AM", "Frecuencia (kHz)", "Amplitud espectral")
    fig.tight_layout()
    fig.savefig(output_dir / "02_espectro_am.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 9. Ruido: comparación temporal.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t[zoom] * 1000, am[zoom], label="AM ideal", linewidth=1.0)
    ax.plot(t[zoom] * 1000, am_noisy[zoom], label="AM con ruido, SNR = 15 dB", linewidth=0.75, alpha=0.8)
    style_axes(ax, "Efecto del ruido aditivo", "Tiempo (ms)", "Amplitud")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_ruido.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 10. Distorsión: recorte temporal.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t[zoom] * 1000, am[zoom], label="AM ideal", linewidth=1.0)
    ax.plot(t[zoom] * 1000, am_clipped[zoom], label=f"AM recortada (límite ±{CLIP_LEVEL:.2f})", linewidth=1.0)
    ax.axhline(CLIP_LEVEL, linestyle="--", alpha=0.65)
    ax.axhline(-CLIP_LEVEL, linestyle="--", alpha=0.65)
    style_axes(ax, "Efecto de la distorsión por recorte", "Tiempo (ms)", "Amplitud")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_distorsion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 11. Atenuación de la señal.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t[zoom] * 1000, am[zoom], label="AM original", linewidth=1.0)
    ax.plot(t[zoom] * 1000, am_attenuated[zoom], label="AM atenuada (50 %)", linewidth=1.0)
    style_axes(ax, "Efecto de la atenuación lineal", "Tiempo (ms)", "Amplitud")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "05_atenuacion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 12. Comparación del mensaje recuperado en banda base.
    display = t <= 0.012
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(t[display] * 1000, message[display], label="Mensaje original", linewidth=2.0)
    for name, estimate in recovered.items():
        ax.plot(t[display] * 1000, estimate[display], label=name, linewidth=1.0, alpha=0.85)
    style_axes(ax, "Comparación del mensaje después de demodular por envolvente", "Tiempo (ms)", "Amplitud normalizada")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "06_mensaje_recuperado.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 13. Métricas complementarias y exportación.
    measured_noise_power = float(np.mean((am_noisy - am) ** 2))
    measured_snr = float(10 * np.log10(signal_power / measured_noise_power))
    input_peak = float(np.max(np.abs(am)))
    carrier_ratio = float(AC / (AC * (1 + MOD_INDEX)))

    summary = {
        "parametros": {
            "frecuencia_muestreo_hz": FS,
            "duracion_s": DURATION,
            "frecuencia_mensaje_hz": FM,
            "frecuencia_portadora_hz": FC,
            "amplitud_portadora": AC,
            "indice_modulacion": MOD_INDEX,
            "snr_ruido_objetivo_db": SNR_DB,
            "ganancia_atenuacion": ATTENUATION,
            "limite_recorte": CLIP_LEVEL,
            "semilla_ruido": RANDOM_SEED,
        },
        "resultados": {
            "snr_ruido_medida_db": measured_snr,
            "porcentaje_muestras_recortadas": clipping_rate,
            "reduccion_potencia_atenuacion_db": power_reduction_db,
            "pico_am_ideal": input_peak,
            "ratio_portadora_a_pico_envolvente": carrier_ratio,
        },
        "metricas_demodulacion": metrics,
    }
    with (output_dir / "resultados.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "metricas_demodulacion.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Escenario", "MSE", "RMSE", "SNR de error (dB)", "Correlación"])
        for name, values in metrics.items():
            writer.writerow([name, values["mse"], values["rmse"], values["snr_error_db"], values["correlacion"]])

    print("Simulación AM terminada.")
    print(f"Resultados guardados en: {output_dir.resolve()}")
    print(f"SNR medida del escenario con ruido: {measured_snr:.2f} dB")
    print(f"Muestras recortadas: {clipping_rate:.2f} %")
    print(f"Reducción de potencia por atenuación: {power_reduction_db:.2f} dB")
    for name, values in metrics.items():
        print(f"{name}: RMSE={values['rmse']:.4f}, correlación={values['correlacion']:.4f}, SNR_error={values['snr_error_db']:.2f} dB")


if __name__ == "__main__":
    main()
