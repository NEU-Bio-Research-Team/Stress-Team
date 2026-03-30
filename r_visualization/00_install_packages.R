# ============================================================
# 00_install_packages.R
# Install all required R packages for the visualization project
# Run this ONCE before running the main scripts
# ============================================================

cat("Installing required packages...\n")

required_packages <- c(
  # Data manipulation
  "tidyverse",     # dplyr, ggplot2, tidyr, readr, purrr, stringr, forcats
  "jsonlite",      # Read JSON validation files
  "data.table",    # Fast CSV reading for large files

  # Visualization
  "ggplot2",       # Core plotting (included in tidyverse but explicit)
  "patchwork",     # Combine multiple ggplots
  "scales",        # Axis formatting
  "ggridges",      # Ridge/joy plots for distributions
  "RColorBrewer",  # Color palettes
  "viridis",       # Perceptually uniform color scales
  "ggbeeswarm",    # Beeswarm plots for subject-level data
  "ggradar",       # Radar/spider charts (optional)
  "corrplot",      # Correlation matrices
  "pheatmap",      # Heatmaps

  # Signal processing visualization
  "signal",        # DSP functions (spectrum, filters)
  "seewave",       # Spectrogram/waveform analysis

  # Statistical
  "effectsize",    # Cohen's d computation
  "broom",         # Tidy model outputs

  # Output
  "Cairo",         # High-quality PDF/PNG rendering
  "svglite"        # SVG output
)

# Install missing packages
installed <- rownames(installed.packages())
to_install <- setdiff(required_packages, installed)

if (length(to_install) > 0) {
  cat(sprintf("Installing %d packages: %s\n", length(to_install), paste(to_install, collapse = ", ")))
  install.packages(to_install, repos = "https://cran.r-project.org", dependencies = TRUE)
} else {
  cat("All packages already installed!\n")
}

# Check ggradar separately (not on CRAN)
if (!"ggradar" %in% installed) {
  cat("Installing ggradar from GitHub...\n")
  if (!"remotes" %in% installed) install.packages("remotes")
  tryCatch(
    remotes::install_github("ricardo-bion/ggradar"),
    error = function(e) cat("  Note: ggradar install failed (optional). Skipping.\n")
  )
}

cat("\nPackage installation complete!\n")
cat("Verifying critical packages...\n")

critical <- c("tidyverse", "jsonlite", "patchwork", "ggridges", "pheatmap", "Cairo")
for (pkg in critical) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  [OK] %s\n", pkg))
  } else {
    cat(sprintf("  [X] %s -- MISSING, please install manually\n", pkg))
  }
}
