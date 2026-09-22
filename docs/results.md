# Benchmark results

[Back to README](../README.md)

The tables below summarize the benchmark results. All values are percentages; higher is better.

## Qwen2.5-3B-Instruct

| Method | ALFWorld Avg. | Search-QA Avg. | WebShop Score | WebShop Success |
| :--- | ---: | ---: | ---: | ---: |
| Vanilla | 19.7 | 31.7 | 6.7 | 0.8 |
| OPSD | 23.3 | 0.0 | 11.3 | 3.1 |
| GRPO | 71.8 | 36.4 | 79.8 | 63.3 |
| GiGPO | 84.1 | 42.6 | 83.8 | 70.3 |
| SDAR | 80.1 | 43.4 | 85.0 | 68.0 |
| StepOPSD | 84.0 | 43.7 | — | — |
| OPID | 84.3 | 45.0 | 85.0 | 74.2 |
| **TASPO** | **86.2** | **46.4** | **88.5** | **78.1** |

## Qwen2.5-7B-Instruct

| Method | ALFWorld Avg. | Search-QA Avg. | WebShop Score | WebShop Success |
| :--- | ---: | ---: | ---: | ---: |
| Vanilla | 10.2 | 33.9 | 5.9 | 1.6 |
| OPSD | 30.2 | 6.2 | 4.5 | 2.3 |
| GRPO | 77.0 | 48.1 | 85.7 | 75.0 |
| GiGPO | 87.5 | 47.1 | 84.4 | 72.8 |
| SDAR | 83.9 | 49.0 | **89.4** | **82.8** |
| OPID | 90.0 | 49.2 | 85.3 | 79.7 |
| **TASPO** | **90.1** | **49.8** | 87.7 | 76.4 |

## Qwen3-1.7B-Instruct

| Method | ALFWorld Avg. | Search-QA Avg. | WebShop Score | WebShop Success |
| :--- | ---: | ---: | ---: | ---: |
| Vanilla | 12.7 | 24.8 | 46.5 | 4.7 |
| OPSD | 13.1 | 5.8 | 47.4 | 9.3 |
| GRPO | 40.1 | 43.6 | 55.8 | 43.0 |
| GiGPO | 64.9 | 41.0 | 70.0 | 45.3 |
| SDAR | 47.6 | 41.9 | 76.8 | 58.6 |
| StepOPSD | 53.9 | 40.7 | — | — |
| OPID | 58.9 | 40.4 | **79.6** | 64.8 |
| **TASPO** | **69.6** | **46.6** | 77.6 | **67.1** |

Bold values mark the best reported result in each column. A dash means the corresponding result is not available in this comparison.

## Evaluation metrics

- **ALFWorld:** macro-average of success rates across Pick, Look, Clean, Heat, Cool, and Pick2. The main table uses the 140-task seen split.
- **Search-QA:** macro-average of exact-match accuracy on NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle. Training uses NQ and HotpotQA.
- **WebShop:** normalized task score and success rate on 128 fixed tasks.

For checkpoint evaluation commands, see the [training guide](usage.md#evaluate).
