# ============================================================
# 02_signal_pipeline_plots.R
# Signal shape visualization: Raw -> Filtered -> Features
# Shows one example case per dataset per preprocessing stage
# ============================================================

library(tidyverse)
library(patchwork)
library(scales)
library(viridis)

# --- Paths ---
BASE_DIR   <- normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."), mustWork = FALSE)
if (!exists("BASE_DIR") || !dir.exists(BASE_DIR)) {
  BASE_DIR <- normalizePath(file.path(getwd(), ".."), mustWork = FALSE)
}
EXPORT_DIR <- file.path(BASE_DIR, "r_visualization", "data_export")
OUTPUT_DIR <- file.path(BASE_DIR, "r_visualization", "output")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=" , rep("=", 58), "\n", sep = "")
cat("Signal Pipeline Plots\n")
cat("=" , rep("=", 58), "\n", sep = "")

# --- Theme ---
theme_publication <- theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0),
    plot.subtitle = element_text(color = "grey30", size = 10, margin = margin(b = 8)),
    strip.text = element_text(face = "bold", size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "grey90"),
    plot.margin = margin(12, 18, 12, 18),
    axis.title = element_text(size = 11),
    legend.text = element_text(size = 10)
  )

# Color palette
STRESS_COLORS <- c("0" = "#2196F3", "1" = "#F44336")
STAGE_COLORS  <- c("Raw" = "#9E9E9E", "Filtered" = "#4CAF50",
                    "Features" = "#FF9800", "Baseline Corrected" = "#9C27B0")

# ============================================================
# PLOT 1: WESAD ECG Pipeline (Raw -> Filtered -> R-peaks -> HRV)
# ============================================================
cat("\n[1/4] WESAD ECG Signal Pipeline...\n")

# Detect subject ID from available files
wesad_raw_files <- list.files(EXPORT_DIR, pattern = "wesad_S\\d+_raw_signals\\.csv")

if (length(wesad_raw_files) > 0) {
  subj_id <- str_extract(wesad_raw_files[1], "S\\d+")

  raw_sig   <- read_csv(file.path(EXPORT_DIR, sprintf("wesad_%s_raw_signals.csv", subj_id)),
                         show_col_types = FALSE)
  filt_sig  <- read_csv(file.path(EXPORT_DIR, sprintf("wesad_%s_filtered_signals.csv", subj_id)),
                         show_col_types = FALSE)
  rr_data   <- read_csv(file.path(EXPORT_DIR, sprintf("wesad_%s_rr_intervals.csv", subj_id)),
                         show_col_types = FALSE)

  # Subset: first 10 seconds (7000 samples at 700 Hz)
  seg <- 7000
  raw_sub  <- raw_sig  %>% filter(sample_idx < seg)
  filt_sub <- filt_sig %>% filter(sample_idx < seg)
  rr_sub   <- rr_data  %>% filter(time_s < seg / 700)

  # Panel A: Raw ECG
  p1a <- ggplot(raw_sub, aes(x = time_s, y = ecg_raw)) +
    geom_line(color = STAGE_COLORS["Raw"], linewidth = 0.3) +
    labs(title = "A. Raw ECG (700 Hz)",
         subtitle = sprintf("%s -- First 10 seconds", subj_id),
         x = "Time (s)", y = "Amplitude (mV)") +
    theme_publication

  # Panel B: Filtered ECG (0.5-40 Hz bandpass)
  p1b <- ggplot(filt_sub, aes(x = time_s, y = ecg_filtered)) +
    geom_line(color = STAGE_COLORS["Filtered"], linewidth = 0.3) +
    labs(title = "B. Filtered ECG (0.5-40 Hz Bandpass)",
         subtitle = "Baseline wander & high-freq noise removed",
         x = "Time (s)", y = "Amplitude (mV)") +
    theme_publication

  # Panel C: R-R intervals
  p1c <- ggplot(rr_sub, aes(x = time_s, y = rr_interval_ms)) +
    geom_point(color = "#E91E63", size = 1.5, alpha = 0.7) +
    geom_line(color = "#E91E63", alpha = 0.4) +
    geom_hline(yintercept = c(250, 2000), linetype = "dashed", color = "grey60") +
    annotate("text", x = max(rr_sub$time_s) * 0.95, y = 280,
             label = "Min (250 ms)", size = 3, color = "grey50") +
    labs(title = "C. R-R Intervals (Pan-Tompkins Detection)",
         subtitle = "Outliers outside 250-2000 ms rejected",
         x = "Time (s)", y = "R-R Interval (ms)") +
    theme_publication

  # Panel D: HRV features across windows (from features CSV)
  wesad_feat <- read_csv(file.path(BASE_DIR, "data", "processed", "wesad_features.csv"),
                          show_col_types = FALSE)
  feat_subj <- wesad_feat %>%
    filter(grepl(subj_id, subject, fixed = TRUE)) %>%
    mutate(time_min = window_idx * 5 / 60,
           label_f = factor(label, levels = c(0, 1), labels = c("Non-stress", "Stress")))

  p1d <- ggplot(feat_subj, aes(x = time_min, y = hr_mean, color = label_f)) +
    geom_point(size = 0.8, alpha = 0.6) +
    geom_smooth(method = "loess", se = FALSE, span = 0.1, linewidth = 0.8) +
    scale_color_manual(values = c("Non-stress" = "#2196F3", "Stress" = "#F44336"),
                       name = "Condition") +
    labs(title = "D. Extracted Feature: Mean Heart Rate",
         subtitle = sprintf("%s -- HRV features per 5-second window", subj_id),
         x = "Time (min)", y = "Heart Rate (bpm)") +
    theme_publication

  # Combine
  p_wesad_ecg <- (p1a | p1b) / (p1c | p1d) +
    plot_annotation(
      title = "WESAD ECG Processing Pipeline",
      subtitle = "Signal transformation from raw chest ECG to HRV features",
      theme = theme(
        plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(color = "grey40", size = 12)
      )
    )

  ggsave(file.path(OUTPUT_DIR, "01_wesad_ecg_pipeline.pdf"),
         p_wesad_ecg, width = 16, height = 10, dpi = 300)
  ggsave(file.path(OUTPUT_DIR, "01_wesad_ecg_pipeline.png"),
         p_wesad_ecg, width = 16, height = 10, dpi = 300)
  cat("  Saved: 01_wesad_ecg_pipeline.pdf/png\n")

} else {
  cat("  SKIP: No raw WESAD signal data found. Run 01_export_data_from_python.py first.\n")
}


# ============================================================
# PLOT 2: WESAD EDA Pipeline (Raw -> Filtered -> Features)
# ============================================================
cat("\n[2/4] WESAD EDA Signal Pipeline...\n")

if (length(wesad_raw_files) > 0) {

  # Panel A: Raw EDA
  p2a <- ggplot(raw_sub, aes(x = time_s, y = eda_raw)) +
    geom_line(color = STAGE_COLORS["Raw"], linewidth = 0.4) +
    labs(title = "A. Raw EDA (700 Hz)",
         subtitle = sprintf("%s -- High-frequency noise visible", subj_id),
         x = "Time (s)", y = expression(paste("Conductance (", mu, "S)"))) +
    theme_publication

  # Panel B: Filtered EDA (5 Hz lowpass)
  p2b <- ggplot(filt_sub, aes(x = time_s, y = eda_filtered)) +
    geom_line(color = STAGE_COLORS["Filtered"], linewidth = 0.5) +
    labs(title = "B. Filtered EDA (5 Hz Lowpass)",
         subtitle = "Smooth skin conductance level (SCL)",
         x = "Time (s)", y = expression(paste("Conductance (", mu, "S)"))) +
    theme_publication

  # Panel C: EDA features over session
  p2c <- ggplot(feat_subj, aes(x = time_min, y = eda_mean, color = label_f)) +
    geom_point(size = 0.8, alpha = 0.6) +
    geom_smooth(method = "loess", se = FALSE, span = 0.1, linewidth = 0.8) +
    scale_color_manual(values = c("Non-stress" = "#2196F3", "Stress" = "#F44336"),
                       name = "Condition") +
    labs(title = "C. Extracted Feature: Mean EDA",
         subtitle = sprintf("%s -- EDA features per 5-second window", subj_id),
         x = "Time (min)", y = expression(paste("EDA Mean (", mu, "S)"))) +
    theme_publication

  # Panel D: EDA slope
  p2d <- ggplot(feat_subj, aes(x = time_min, y = eda_slope, color = label_f)) +
    geom_point(size = 0.8, alpha = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    scale_color_manual(values = c("Non-stress" = "#2196F3", "Stress" = "#F44336"),
                       name = "Condition") +
    labs(title = "D. Extracted Feature: EDA Slope",
         subtitle = "Positive slope = rising sympathetic activation",
         x = "Time (min)", y = expression(paste("Slope (", mu, "S/s)"))) +
    theme_publication

  p_wesad_eda <- (p2a | p2b) / (p2c | p2d) +
    plot_annotation(
      title = "WESAD EDA Processing Pipeline",
      subtitle = "Signal transformation from raw electrodermal activity to stress features",
      theme = theme(
        plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(color = "grey40", size = 12)
      )
    )

  ggsave(file.path(OUTPUT_DIR, "02_wesad_eda_pipeline.pdf"),
         p_wesad_eda, width = 16, height = 10, dpi = 300)
  ggsave(file.path(OUTPUT_DIR, "02_wesad_eda_pipeline.png"),
         p_wesad_eda, width = 16, height = 10, dpi = 300)
  cat("  Saved: 02_wesad_eda_pipeline.pdf/png\n")

} else {
  cat("  SKIP: No raw WESAD data. Run Python export first.\n")
}


# ============================================================
# PLOT 3: DREAMER EEG Pipeline (Raw -> Filtered -> Baseline -> DE)
# ============================================================
cat("\n[3/4] DREAMER EEG Signal Pipeline...\n")

dreamer_raw_files <- list.files(EXPORT_DIR, pattern = "dreamer_S\\d+_raw_eeg\\.csv")

if (length(dreamer_raw_files) > 0) {
  subj_dr <- str_extract(dreamer_raw_files[1], "S\\d+")

  raw_eeg  <- read_csv(file.path(EXPORT_DIR, sprintf("dreamer_%s_raw_eeg.csv", subj_dr)),
                         show_col_types = FALSE)
  filt_eeg <- read_csv(file.path(EXPORT_DIR, sprintf("dreamer_%s_filtered_eeg.csv", subj_dr)),
                         show_col_types = FALSE)
  corr_eeg <- read_csv(file.path(EXPORT_DIR, sprintf("dreamer_%s_baseline_corrected_eeg.csv", subj_dr)),
                         show_col_types = FALSE)

  # Select 2 representative channels: AF3 (frontal) and O1 (occipital)
  chan_show <- c("AF3", "O1")

  # Reshape for faceted plot
  raw_long <- raw_eeg %>%
    select(time_s, all_of(chan_show)) %>%
    pivot_longer(-time_s, names_to = "channel", values_to = "amplitude") %>%
    mutate(stage = "1. Raw EEG")

  filt_long <- filt_eeg %>%
    select(time_s, all_of(chan_show)) %>%
    pivot_longer(-time_s, names_to = "channel", values_to = "amplitude") %>%
    mutate(stage = "2. Bandpass + Notch Filtered")

  corr_long <- corr_eeg %>%
    select(time_s, all_of(chan_show)) %>%
    pivot_longer(-time_s, names_to = "channel", values_to = "amplitude") %>%
    mutate(stage = "3. Baseline Subtracted")

  all_eeg <- bind_rows(raw_long, filt_long, corr_long)

  p3_eeg <- ggplot(all_eeg, aes(x = time_s, y = amplitude, color = stage)) +
    geom_line(linewidth = 0.25) +
    facet_grid(channel ~ stage, scales = "free_y") +
    scale_color_manual(values = c("1. Raw EEG" = "#9E9E9E",
                                  "2. Bandpass + Notch Filtered" = "#4CAF50",
                                  "3. Baseline Subtracted" = "#9C27B0")) +
    labs(title = "DREAMER EEG Processing Pipeline",
         subtitle = sprintf("%s Trial 1 -- Channels AF3 (frontal) & O1 (occipital), 10 seconds at 128 Hz", subj_dr),
         x = "Time (s)", y = expression(paste("Amplitude (", mu, "V)"))) +
    theme_publication +
    theme(legend.position = "none",
          strip.text = element_text(size = 9))

  ggsave(file.path(OUTPUT_DIR, "03_dreamer_eeg_pipeline.pdf"),
         p3_eeg, width = 16, height = 8, dpi = 300)
  ggsave(file.path(OUTPUT_DIR, "03_dreamer_eeg_pipeline.png"),
         p3_eeg, width = 16, height = 8, dpi = 300)
  cat("  Saved: 03_dreamer_eeg_pipeline.pdf/png\n")

} else {
  cat("  SKIP: No raw DREAMER EEG data. Run Python export first.\n")
}


# ============================================================
# PLOT 4: DREAMER Differential Entropy Feature Space
# ============================================================
cat("\n[4/4] DREAMER DE Feature Heatmap...\n")

dreamer_de_files <- list.files(EXPORT_DIR, pattern = "dreamer_S\\d+_de_features\\.csv")

if (length(dreamer_de_files) > 0) {
  de_data <- read_csv(file.path(EXPORT_DIR, dreamer_de_files[1]), show_col_types = FALSE)

  channels <- c("AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                 "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")
  bands <- c("delta", "theta", "alpha", "beta", "gamma")

  # Compute mean DE per channel x band (first 100 windows)
  de_matrix <- de_data %>%
    slice(1:min(100, n())) %>%
    summarise(across(everything(), mean)) %>%
    pivot_longer(everything(), names_to = "feature", values_to = "de_value") %>%
    mutate(
      idx = as.integer(str_extract(feature, "\\d+")),
      channel = channels[(idx %% 14) + 1],
      band = bands[(idx %/% 14) + 1]
    ) %>%
    filter(!is.na(channel), !is.na(band))

  # Heatmap
  p4 <- ggplot(de_matrix, aes(x = band, y = channel, fill = de_value)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = round(de_value, 2)), size = 2.8) +
    scale_fill_viridis(option = "magma", name = "DE Value") +
    scale_x_discrete(limits = bands) +
    scale_y_discrete(limits = rev(channels)) +
    labs(title = "DREAMER: Differential Entropy Heatmap",
         subtitle = sprintf("%s -- Mean DE per channel x frequency band (first 100 windows)", subj_dr),
         x = "Frequency Band", y = "EEG Channel") +
    theme_publication +
    theme(axis.text.x = element_text(angle = 0),
          panel.grid = element_blank())

  ggsave(file.path(OUTPUT_DIR, "04_dreamer_de_heatmap.pdf"),
         p4, width = 10, height = 8, dpi = 300)
  ggsave(file.path(OUTPUT_DIR, "04_dreamer_de_heatmap.png"),
         p4, width = 10, height = 8, dpi = 300)
  cat("  Saved: 04_dreamer_de_heatmap.pdf/png\n")

} else {
  cat("  SKIP: No DREAMER DE features found.\n")
}

cat("\nSignal pipeline plots complete!\n")
