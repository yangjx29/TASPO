# Contributing

Thanks for your interest in TASPO. Bug reports, fixes, and clearer documentation are welcome.

For a bug report, include the command you ran, the relevant configuration, Python/PyTorch/vLLM versions, GPU model, and the error log. Remove API keys and other credentials from logs before sharing them.

For code changes, describe the behavior being changed and how you checked it. Run the core tests from the repository root:

```bash
bash scripts/taspo/run_unit_tests.sh
```

Changes to training or environment integration should also include the relevant smoke-test output when GPU resources are available. Keep third-party copyright notices intact.
