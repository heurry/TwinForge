# CPT Communication Report

本报告优先使用 `torch profiler` 观察 DDP/ZeRO-2 的通信热点，并补充 DeepSpeed comms logger 的聚合结果。`zero3_offload` 当前仅纳入 DeepSpeed comms logger，因为该 profile 路线已关闭 torch profiler 以保证稳定完成。

| Backend | Source | Total Comm Time (us) | Comm Share | Top Comm Op |
| --- | --- | ---: | ---: | --- |
| ddp | torch profiler | 2441.8 | 0.0 | c10d::allreduce_ |
| zero2 | torch profiler | 17051.1 | 0.0 | c10d::allreduce_ |
| zero2 | deepspeed comms logger | - | - | all_reduce |
| zero3_offload | deepspeed comms logger | - | - | all_gather_into_tensor |

## Analysis

- DDP：torch profiler 中显式通信事件以 `c10d::allreduce_` 为主，但总通信时间占比约 `0.0%`，说明 native 路线的主要瓶颈并不在显式 NCCL 通信，而更可能在 kernel launch、同步和运行时调度。
- ZeRO-2：torch profiler 中 top comm op 也是 `c10d::allreduce_`，而 DeepSpeed comms logger 里按聚合延迟看则明显由 `all_reduce` 主导；这说明 `zero2` 仍以经典梯度归约为主，但其真实通信成本更适合用 DeepSpeed comms logger 的 op mix 来理解。
- ZeRO-2 vs DDP：torch profiler 统计到的显式通信时间从 `2441.8` 增到 `17051.1`，约 `6.98x`；不过两边 share 都接近 `0%`，因此这份报告更适合判断通信形态，而不是单独用 share 解释全部性能差异。
- ZeRO-3 offload：DeepSpeed comms logger 显示其通信不再由 `all_reduce` 主导，而是 `all_gather_into_tensor` + `reduce_scatter_tensor` 主导。这和 ZeRO-3 CPU offload 的机制一致，说明它的主要代价是参数/梯度分片的 gather/scatter 与 host-device 搬运，而不是普通 DDP 式的梯度归约。
- Interpretation：后续如果目标是提高 `ddp` 吞吐，不该优先把时间花在 NCCL 调参上；如果目标是理解 `zero3_offload` 为什么慢，这份报告已经足够支持“它慢在分片搬运与 offload orchestration，而不是慢在 classic all-reduce”这个判断。

## DDP Torch Profiler Communication Ops

- `c10d::allreduce_` count=`171` time=`1905.7`us share=`0.0`
- `c10d::_allgather_base_` count=`15` time=`386.4`us share=`0.0`
- `c10d::allgather_` count=`6` time=`114.6`us share=`0.0`
- `c10d::broadcast_` count=`3` time=`35.1`us share=`0.0`
- `nccl:all_gather` count=`6` time=`0.0`us share=`0.0`
- `nccl:all_gather` count=`6` time=`0.0`us share=`0.0`
- `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` count=`21` time=`0.0`us share=`0.0`
- `nccl:_all_gather_base` count=`15` time=`0.0`us share=`0.0`
- `nccl:_all_gather_base` count=`15` time=`0.0`us share=`0.0`
- `nccl:broadcast` count=`3` time=`0.0`us share=`0.0`

## ZeRO-2 Torch Profiler Communication Ops

- `c10d::allreduce_` count=`823` time=`15628.6`us share=`0.0`
- `c10d::_allgather_base_` count=`31` time=`1351.6`us share=`0.0`
- `c10d::allgather_` count=`6` time=`71.0`us share=`0.0`
- `nccl:all_gather` count=`6` time=`0.0`us share=`0.0`
- `nccl:all_gather` count=`6` time=`0.0`us share=`0.0`
- `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` count=`37` time=`0.0`us share=`0.0`
- `nccl:_all_gather_base` count=`31` time=`0.0`us share=`0.0`
- `nccl:_all_gather_base` count=`31` time=`0.0`us share=`0.0`
- `nccl:all_reduce` count=`823` time=`0.0`us share=`0.0`
- `ncclDevKernel_AllReduce_Sum_f16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` count=`816` time=`0.0`us share=`0.0`

## ZeRO-2 DeepSpeed Comms Logger Summary

- `all_reduce` count=`3294` total_latency=`143082918.439`ms
- `broadcast` count=`310` total_latency=`121827.662`ms
- `all_gather_into_tensor` count=`18` total_latency=`48050.885`ms
- `log_summary_barrier` count=`1` total_latency=`0.799`ms

## ZeRO-3 Offload DeepSpeed Comms Logger Summary

- `all_gather_into_tensor` count=`8644` total_latency=`169670856.253`ms
- `reduce_scatter_tensor` count=`8128` total_latency=`105433633.157`ms
- `broadcast` count=`628` total_latency=`422313.605`ms
- `all_reduce` count=`6` total_latency=`1428.626`ms
- `barrier` count=`3` total_latency=`16.294`ms
- `log_summary_barrier` count=`1` total_latency=`0.094`ms
