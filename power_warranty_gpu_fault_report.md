# Warranty Service Fault Report: Acer Predator PO5-650 RTX 4080

This document summarizes the GPU stability fault observed on the desktop PC for POWER warranty/service handling.

## Device And Receipt Information

- Product: Acer Predator PO5-650 desktop PC
- Receipt product description: `ACER PO5-650 I7/16/1TB/4080`
- GPU: NVIDIA GeForce RTX 4080, 16 GB VRAM
- Original purchase store: POWER Tammisto
- Original purchase date: 19 Oct 2025
- Warranty number on receipt: `202506601794`
- Receipt/order references visible on receipt: `XQ211X6`, `4725 XV502V6`
- Current owner: bought second-hand from the original buyer; original POWER receipt copy is available.

## Short Fault Description

The PC is unstable under sustained GPU compute load. The machine can boot and show normal desktop output, but during CUDA/GPU workloads the NVIDIA driver loses communication with the RTX 4080. After this happens, `nvidia-smi` can no longer access the GPU and the machine requires reboot or power-cycle to recover GPU access.

The failure is not normal application failure. The kernel and NVIDIA driver report that the GPU has fallen off the PCIe bus.

Most important observed errors:

- `NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.`
- `NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)`
- `NVRM: GPU lost from the bus [NV_ERR_GPU_IS_LOST]`
- `CUDA error: unspecified launch failure`
- `nvidia-smi`: `Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown Error`
- `nvidia-smi`: `No devices were found`

The issue appears under sustained GPU/CUDA/LLM workload. It is not limited to a specific Python script, because the failure happens at the NVIDIA driver / PCIe / GPU-device level.

## Important Observation About GPU Power Limit

At the normal/default GPU power limit, the machine has crashed under GPU load. The GPU default power limit reported by `nvidia-smi` is `320 W`.

The system became more stable after manually limiting the GPU to the minimum allowed power limit of `150 W`:

```text
Current Power Limit:   150.00 W
Requested Power Limit: 150.00 W
Default Power Limit:   320.00 W
Min Power Limit:       150.00 W
Max Power Limit:       320.00 W
```

This 150 W limit is only a workaround. The PC should be stable at its normal factory/default GPU settings. The fact that it becomes more stable only after artificially reducing RTX 4080 power suggests a possible GPU, PSU, PCIe, motherboard, power-delivery, or driver/firmware stability issue.

## System / Driver Information At Time Of Testing

From `nvidia-smi` and NVIDIA bug report:

```text
NVIDIA-SMI version: 595.71.05
Driver version:     595.71.05
CUDA version:       13.2
GPU:                NVIDIA GeForce RTX 4080
Bus:                0000:01:00.0
```

Linux host:

```text
Hostname: isaac-Predator-PO5-650
Product Name from NVIDIA bug report: Predator PO5-650
Motherboard/Product Name from bug report: H77H6-AM
```

## Exact NVIDIA Kernel Error Logs

The following lines were captured from the NVIDIA bug reports and kernel logs on 1 Jun 2026.

### Crash Event: 1 Jun 2026, 15:55

```text
2026-06-01T15:55:06.218036+03:00 isaac-Predator-PO5-650 kernel: NVRM: Xid (PCI:0000:01:00): 79, pid=30393, name=Media, GPU has fallen off the bus.
2026-06-01T15:55:06.218037+03:00 isaac-Predator-PO5-650 kernel: NVRM: GPU 0000:01:00.0: GPU has fallen off the bus.
2026-06-01T15:55:06.218447+03:00 isaac-Predator-PO5-650 kernel: NVRM: nvCheckOkFailedNoLog: Check failed: GPU lost from the bus [NV_ERR_GPU_IS_LOST] (0x0000000F)
2026-06-01T15:55:06.218530+03:00 isaac-Predator-PO5-650 kernel: NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
```

### Crash Event: 1 Jun 2026, 17:12

```text
Jun 01 17:12:22 isaac-Predator-PO5-650 kernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.
Jun 01 17:12:22 isaac-Predator-PO5-650 kernel: NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
```

### Crash Event: 1 Jun 2026, 20:11

```text
Jun 01 20:11:11 isaac-Predator-PO5-650 kernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.
Jun 01 20:11:11 isaac-Predator-PO5-650 kernel: NVRM: GPU 0000:01:00.0: GPU has fallen off the bus.
Jun 01 20:11:11 isaac-Predator-PO5-650 kernel: NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
Jun 01 20:11:11 isaac-Predator-PO5-650 kernel: NVRM: nvCheckOkFailedNoLog: Check failed: GPU lost from the bus [NV_ERR_GPU_IS_LOST] (0x0000000F)
```

### Crash Event: 1 Jun 2026, 20:40

```text
Jun 01 20:40:25 isaac-Predator-PO5-650 kernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.
Jun 01 20:40:25 isaac-Predator-PO5-650 kernel: NVRM: GPU 0000:01:00.0: GPU has fallen off the bus.
Jun 01 20:40:25 isaac-Predator-PO5-650 kernel: NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
Jun 01 20:40:25 isaac-Predator-PO5-650 kernel: NVRM: nvCheckOkFailedNoLog: Check failed: GPU lost from the bus [NV_ERR_GPU_IS_LOST] (0x0000000F)
```

## `nvidia-smi` Failure After Crash

After the GPU falls off the bus, NVIDIA tools cannot access the GPU:

```text
NVIDIA GPU Details | Failed: Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown Error
Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown Error
No devices were found
```

This is why the machine must be rebooted/power-cycled before the GPU becomes visible again.

## CUDA / vLLM Runtime Failure

During a GPU inference workload, the model server failed with a CUDA launch failure:

```text
torch.AcceleratorError: CUDA error: unspecified launch failure
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
```

After this error, the API server returned repeated internal server errors and the evaluation could no longer use the GPU.

## Metrics From Failed Runs

The monitoring logs show that the crashes were not caused by overheating or full VRAM usage.

### Failed run at default/uncapped GPU power limit

Metrics file:

```text
outputs/system_metrics/student_eval_qwen_0_8b_vllm_5120_fixed_20260601_200955.csv
```

Summary:

```text
Power limit:       280 W during this run
Max observed draw: 199.68 W
Max temperature:   46 C
Max GPU memory:    8468 MiB / 16376 MiB
Result:            GPU disappeared; later samples reported "No devices were found"
```

Example log tail after failure:

```text
2026-06-01T20:16:28+03:00,...,Nodeviceswerefound
2026-06-01T20:16:33+03:00,...,Nodeviceswerefound
2026-06-01T20:16:38+03:00,...,Nodeviceswerefound
2026-06-01T20:16:43+03:00,...,Nodeviceswerefound
```

### Failed run with GPU power limit reduced to 200 W

Metrics file:

```text
outputs/system_metrics/student_qwen08b_5120_eager_20260601_203637.csv
```

Summary:

```text
Power limit:       200 W
Max observed draw: 118.29 W
Max temperature:   46 C
Max GPU memory:    6639 MiB / 16376 MiB
Result:            GPU disappeared; later samples reported "No devices were found"
```

Example log tail after failure:

```text
No devices were found
no_gpu,no_gpu,no_gpu,no_gpu,no_gpu,no_gpu,no_gpu,...
No devices were found
no_gpu,no_gpu,no_gpu,no_gpu,no_gpu,no_gpu,no_gpu,...
```

This is important: the failure occurred even when the sampled power draw was far below the 200 W cap and the GPU temperature was only about 46 C.

## More Stable Run With 150 W Cap

After reducing the RTX 4080 power limit to the minimum allowed value, `150 W`, the same class of workload was more stable.

Metrics file:

```text
outputs/system_metrics/student_qwen08b_vllm_150w_20260601_205235.csv
```

Summary:

```text
Power limit:       150 W
Max observed draw: 149.56 W
Max temperature:   51 C
Max GPU memory:    6913 MiB / 16376 MiB
No "No devices were found" rows in this monitored run.
```

Current observed training state under 150 W cap:

```text
NVIDIA-SMI 595.71.05
GPU: NVIDIA GeForce RTX 4080
Power: 128 W / 150 W
Temperature: 52 C
Memory: 5267 MiB / 16376 MiB
GPU Utilization: 61%
```

This suggests the machine may be more stable only when the GPU is artificially power-limited, but this should not be required for a desktop PC sold with an RTX 4080.

## Requested Service Action

Please diagnose and repair the PC under warranty. The requested diagnosis is specifically GPU stability under load, not only whether the desktop display works.

Please test:

- Sustained GPU/CUDA/3D load at default factory GPU power settings.
- Whether the RTX 4080 falls off the PCIe bus.
- GPU power delivery, PSU/cabling, PCIe slot/motherboard stability, and GPU hardware.
- NVIDIA driver logs for `Xid 79`, `Xid 154`, `NV_ERR_GPU_IS_LOST`, and `Unknown Error`.

The customer-observed workaround of limiting the GPU to 150 W is not a final fix. The machine should be stable at normal factory settings.

## Files Available If Needed

The following local logs were collected during diagnosis on the machine:

```text
/home/isaac/gpu-crash-logs/bug-reports/nvidia-bug-report-after-xid79-20260601_201930.log.gz.gz
/home/isaac/gpu-crash-logs/bug-reports/nvidia-bug-report-after-xid79-20260601_204147.log.gz.gz
/home/isaac/gpu-crash-logs/vllm/student_qwen08b_5120_eager_20260601_203637.log
/home/isaac/gpu-crash-logs/vllm/student_qwen08b_5120_eager_20260601_203637_eval.log
/home/isaac/codes/personal/small-lm-training/outputs/system_metrics/student_eval_qwen_0_8b_vllm_5120_fixed_20260601_200955.csv
/home/isaac/codes/personal/small-lm-training/outputs/system_metrics/student_qwen08b_5120_eager_20260601_203637.csv
/home/isaac/codes/personal/small-lm-training/outputs/system_metrics/student_qwen08b_vllm_150w_20260601_205235.csv
```

