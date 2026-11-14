Set-Location 'C:\NAQSHKASH'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH='C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
& $Python312 -B -m pytest stage2\preprocessing_improvements\tests tests\test_preprocessing_rows.py tests\test_preprocessing_crop.py tests\test_preprocessing_pipeline.py tests\test_cleaning.py tests\test_ruling_lines.py tests\test_crnn.py tests\test_training_numerics.py::test_runtime_ctc_repeats_and_actual_widths tests\test_training_numerics.py::test_ctc_receives_fp32_even_under_autocast -q -p no:cacheprovider --basetemp stage2\preprocessing_improvements\reports\pytest_temp_v1 --junitxml stage2\preprocessing_improvements\reports\tests.xml 2>&1 | Tee-Object -FilePath stage2\preprocessing_improvements\reports\tests.txt
exit $LASTEXITCODE
