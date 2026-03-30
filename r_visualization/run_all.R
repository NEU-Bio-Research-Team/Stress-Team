# ============================================================
# run_all.R
# Master script: Run ALL visualization scripts in sequence
# Just open this file in RStudio and click "Source" (Ctrl+Shift+S)
# ============================================================

cat("\n")
cat("##########################################################\n")
cat("#  Algorithmic Panic -- Bio Stage Visualization Pipeline #\n")
cat("#  R Visualization Project                                #\n")
cat("##########################################################\n\n")

# Set working directory to this script's location
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
  # Fallback: try rstudioapi
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    script_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
  } else {
    stop("Cannot determine script directory. Please setwd() to the r_visualization folder manually.")
  }
}
setwd(script_dir)
cat(sprintf("Working directory: %s\n\n", getwd()))

# --- Step 0: Install packages if needed ---
cat("Step 0: Checking packages...\n")
source("00_install_packages.R", local = TRUE)
cat("\n")

# --- Step 1: Check exported data ---
export_dir <- file.path(script_dir, "data_export")
export_files <- list.files(export_dir, pattern = "\\.csv$")

if (length(export_files) < 3) {
  cat("Step 1: Data export NOT found.\n")
  cat("  Please run the Python script FIRST (outside RStudio):\n")
  cat("    cd r_visualization\n")
  cat("    python 01_export_data_from_python.py\n")
  cat("  Then re-run this R script.\n")
  cat("  (Signal pipeline plots 01-04 need exported data; contribution plots 05-18 will still work.)\n\n")
} else {
  cat(sprintf("Step 1: Data already exported (%d CSV files found). OK.\n\n",
              length(export_files)))
}

# --- Step 2: Signal Pipeline Plots ---
cat("Step 2: Generating signal pipeline plots...\n")
tryCatch({
  source("02_signal_pipeline_plots.R", local = TRUE)
}, error = function(e) {
  cat(sprintf("  ERROR in signal plots: %s\n", e$message))
  cat("  Continuing to next step...\n")
})
cat("\n")

# --- Step 3: Contribution Plots ---
cat("Step 3: Generating contribution & validation plots...\n")
tryCatch({
  source("03_contribution_plots.R", local = TRUE)
}, error = function(e) {
  cat(sprintf("  ERROR in contribution plots: %s\n", e$message))
})
cat("\n")

# --- Summary ---
output_dir <- file.path(script_dir, "output")
output_files <- list.files(output_dir, pattern = "\\.(pdf|png)$")
pdf_count <- sum(grepl("\\.pdf$", output_files))
png_count <- sum(grepl("\\.png$", output_files))

cat("##########################################################\n")
cat("#  PIPELINE COMPLETE                                     #\n")
cat("##########################################################\n\n")
cat(sprintf("Output: %s\n", output_dir))
cat(sprintf("Generated: %d PDF files + %d PNG files\n\n", pdf_count, png_count))

cat("Plot Inventory:\n")
cat("-----------------------------------------------------------\n")
cat("Signal Pipeline (Task 1):\n")
cat("  01 - WESAD ECG Pipeline (Raw -> Filter -> R-peaks -> HRV)\n")
cat("  02 - WESAD EDA Pipeline (Raw -> Filter -> Features)\n")
cat("  03 - DREAMER EEG Pipeline (Raw -> Filter -> Baseline -> DE)\n")
cat("  04 - DREAMER DE Feature Heatmap\n")
cat("\nContribution Plots (Task 2):\n")
cat("  05 - C1: Stress Detectability (ridges + effect sizes)\n")
cat("  06 - C2: Adversarial Robustness (GRL bar charts)\n")
cat("  07 - C3: Model Comparison (LogReg > CNN)\n")
cat("  08 - C4: Feature Ablation Hierarchy\n")
cat("  09 - C5+C6: OU Process (simulated paths + theta scaling)\n")
cat("  10 - C7+C8: Negative Control & Noise Ceiling\n")
cat("  11 - C9: BTC Stylized Facts Summary\n")
cat("  12 - C10: Representation Transfer (CKA = 0)\n")
cat("  13 - F1: Deep Learning vs Baseline (falsified)\n")
cat("  14 - Subject-Level Variability (lollipop)\n")
cat("  15 - DREAMER Recovery Strategies\n")
cat("  16 - WESAD Feature Correlation Matrix\n")
cat("  17 - Results Dashboard (comprehensive)\n")
cat("  18 - Data Audit Summary\n")
cat("-----------------------------------------------------------\n")

cat("\nAll plots saved in both PDF (vector) and PNG (raster) formats.\n")
cat("Done!\n")
