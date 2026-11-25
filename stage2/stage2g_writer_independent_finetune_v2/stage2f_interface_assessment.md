# Stage 2F rejection-policy interface assessment

Status: NOT RUN, not a passing regression result.

Inspected existing read-only interfaces:
- stage2/stage2f_negative_rejection/src/evaluate_baseline.py: load_recognizer(device) checks and loads the single frozen E1 checkpoint; FrozenPipeline.__init__(device) exposes no checkpoint or normalization argument. observe(raw) hard-codes original aspect-preserving tier preprocessing.
- stage2/stage2f_rejection_evaluation_v2/src/integration.py: make_pixel_pipeline(device) instantiates that frozen pipeline; CHECKPOINT_SHA is fixed to original E1.
- stage2/stage2f_rejection_evaluation_v2/src/evaluate_rejection.py: load_candidate verifies the original frozen checkpoint SHA and original raw-run lineage. Its candidate choices R0/R1 refer to rejection/interpretation policy options, not interchangeable OCR checkpoints.
- r3_hitl_v3_1_targeted_fix/src/paths_v3_1.py and policy_v3_1.py: policy consumes complete aligned page observations and pixel measurements tied to decoded rows and frozen D1 coordinates, not isolated tier predictions.

The current experiment supplies isolated tier inference with two actual preprocessing pipelines. There is no supported checkpoint-plus-role-aware-normalization injection through these existing interfaces. Monkey-patching the model, replacing runtime normalization or fabricating page observations would bypass existing lineage checks or require a new adapter. Under the user's conditional scope this is left unassessed. No rejection policy is retrained, retuned, edited or claimed preserved by the OCR CER checks. A candidate passing OCR gates would still need a separately authorized end-to-end rejection-policy regression before deployment.
