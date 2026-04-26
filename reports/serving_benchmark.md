# Serving Benchmark

最小可运行 serving benchmark，当前后端：`qwen3-1.7b-miniv2`。

| Request | TTFT (s) | End-to-End (s) | Output Tokens | Decode Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.038 | 0.209 | 37 | 216.322 |
| 2 | 0.013 | 0.817 | 161 | 200.024 |
| 3 | 0.011 | 1.138 | 255 | 226.233 |

## Summary

- mean TTFT: `0.020s`
- mean end-to-end latency: `0.722s`
- mean decode tokens/s: `214.193`
