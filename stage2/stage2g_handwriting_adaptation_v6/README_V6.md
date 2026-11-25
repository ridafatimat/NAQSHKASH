# NAQSHKASH V6: retrain with the new collection-v3 writers

Folder: C:\NAQSHKASH\stage2\stage2g_handwriting_adaptation_v6

1. Extract this ZIP into that folder (run_v6.py, config_v6.json, collection_v3_inputs.csv).
2. Extract naqshkash_collection_v3_tier_crops.zip into the SAME folder, so you get
   C:\NAQSHKASH\stage2\stage2g_handwriting_adaptation_v6\handwritten_tier_crops\tier_crops\...
3. In a VS Code terminal:
     cd C:\NAQSHKASH\stage2\stage2g_handwriting_adaptation_v6
     python -B run_v6.py --check
   Expected: CHECK OK train tiers 2627 writers 17 | validation tiers 65 writers ['W05','W24','W30','W31']
4. Then:
     python -B run_v6.py
   If it stops (sleep, Ctrl+C): python -B run_v6.py --resume

What it does
- Starts from the V5b final model (sha 0dc250f1...).
- Trains on W01 W03 W04 W06 + 13 new writers; validation W05 + W24 W30 W31 picks the best checkpoint.
- 1000-3000 updates, evaluation every 250, stops after 4 evaluations without improvement.
- At the end it compares frozen E1, V5b final and V6 on the same 65 validation tiers and on clean synthetic,
  and writes outputs\v6_run\v6_summary.md and v6_summary.json.
- Test writers (W02 W07 W14 W17 W20 W25 W27 W28) are never loaded. No V5b file is modified.
