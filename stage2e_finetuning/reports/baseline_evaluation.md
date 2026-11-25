# Original checkpoint E0 baseline

{
  "clean_historical_regression": {
    "tiers": 645,
    "parents": 98,
    "characters": 4055,
    "cer": 0.0017262638717632551,
    "exact_accuracy": 0.9906976744186047,
    "space_normalized_diagnostic_cer": 0.001429388221841052,
    "macro_parent_cer": 0.002300069341069164,
    "substitutions": 5,
    "deletions": 2,
    "insertions": 0,
    "ctc_loss": 0.013055518279714644
  },
  "clean_validation": {
    "tiers": 63,
    "parents": 60,
    "characters": 380,
    "cer": 0.005263157894736842,
    "exact_accuracy": 0.9682539682539683,
    "space_normalized_diagnostic_cer": 0.006060606060606061,
    "macro_parent_cer": 0.019047619047619046,
    "substitutions": 2,
    "deletions": 0,
    "insertions": 0,
    "ctc_loss": 0.025682418407035987
  },
  "positive_development": {
    "tiers": 431,
    "parents": 46,
    "characters": 2950,
    "cer": 0.12305084745762712,
    "exact_accuracy": 0.9373549883990719,
    "space_normalized_diagnostic_cer": 0.01145374449339207,
    "macro_parent_cer": 0.03174262888972651,
    "substitutions": 17,
    "deletions": 346,
    "insertions": 0,
    "ctc_loss": 0.13118522941333347
  },
  "scanned_exploratory": {
    "tiers": 1354,
    "parents": 50,
    "characters": 8039,
    "cer": 0.020524940913048885,
    "exact_accuracy": 0.9254062038404727,
    "space_normalized_diagnostic_cer": 0.020219279510180835,
    "macro_parent_cer": 0.02572576061213066,
    "substitutions": 120,
    "deletions": 45,
    "insertions": 0,
    "ctc_loss": 0.1511477116584004
  },
  "long_positive": {
    "tiers": 9,
    "parents": 1,
    "characters": 2274,
    "cer": 0.15083553210202286,
    "exact_accuracy": 0.2222222222222222,
    "space_normalized_diagnostic_cer": 0.004369538077403246,
    "macro_parent_cer": 0.15083553210202286,
    "substitutions": 0,
    "deletions": 343,
    "insertions": 0,
    "ctc_loss": 0.08293506744141471
  },
  "short_positive": {
    "tiers": 422,
    "parents": 45,
    "characters": 676,
    "cer": 0.029585798816568046,
    "exact_accuracy": 0.95260663507109,
    "space_normalized_diagnostic_cer": 0.02844311377245509,
    "macro_parent_cer": 0.029096119929453265,
    "substitutions": 17,
    "deletions": 3,
    "insertions": 0,
    "ctc_loss": 0.13221426130372985
  },
  "runtime": {
    "seconds": 298.96752320000087,
    "unique_model_inputs": 2484,
    "cached_tiers": 9,
    "device": "cpu",
    "average_uncached_seconds": 0.11986363289046298
  }
}

These are fixed parent-disjoint exposed development/regression inputs, not an independent test. Full page-level segmentation rejections/wrong acceptances are retained in manifests/page_outcomes.csv; CER is conditional on independently correct frozen D1 crops. No hard evaluable tier is dropped or truncated. Space-normalized CER is diagnostic only.
