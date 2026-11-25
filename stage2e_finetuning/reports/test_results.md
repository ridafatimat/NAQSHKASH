# Actual Stage2E test results

{
  "tests": 41,
  "failures": 0,
  "errors": 0,
  "skipped": 0,
  "passed": 41,
  "seconds": 15.701
}

Focused contracts passed 14/14 before fine-tuning. The final suite includes original checkpoint/vocabulary compatibility, repeated-character CTC feasibility, actual normalization, loader behavior, exact-input and parent isolation, deterministic mixed sampling, real optimization/reload evidence, deterministic next-update optimizer/RNG resume, paired fixed inputs, metrics, clean-regression/model-selection gates and complete historical hash protection, plus relevant unchanged Stage1 CRNN/vocabulary/CTC tests. No tests or failures are silently omitted.

See test_command.ps1, focused_tests.txt, tests.txt and tests.xml. No CUDA execution was tested because the installed stack is CPU-only.
