# Blend two score matrices (rows: home/away, cols: sims)
blend_scores <- function(mat_pbp, mat_eckel, w_eckel = 0.5) {
  w <- min(max(w_eckel, 0), 1)
  (1 - w) * mat_pbp + w * mat_eckel
}

# convenience: choose weight by week (if you want)
w_by_week <- function(week) {
  # e.g., slightly higher Eckel first two weeks, then taper
  if (week <= 2) return(0.55)
  if (week == 3) return(0.52)
  0.50
}
