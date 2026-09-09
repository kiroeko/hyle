# Public C++ inference-service vertical slice

This is a consumer probe, not an inference engine or a web framework. It uses
only public HYLE C++ product targets to exercise one coherent journey:

- Platform chooses a quota-aware worker ceiling and reports host memory;
- Mem optionally maps a native-float model file;
- Util publishes and selects a mock backend;
- Sched owns workers and cancellation;
- IO owns TCP and HTTP/1 framing;
- SIMD evaluates the fake model with `dot`;
- Obs supplies concurrent metric values;
- HTTP exposes `/health`, `/v1/models`, `/v1/infer`, `/v1/stream`, and
  `/metrics`.

The default model is eight demo weights. Start the server on the project-wide
default port with metrics enabled:

```bash
cmake -S samples/08_inference_service -B build/inference-service -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/inference-service --parallel 2
build/inference-service/hyle_inference_service --port=2048 --metrics
```

On Windows the executable may be under `Release/` for a multi-config generator.
Use `--model=/path/to/native-f32.bin` for a mapped float array. This is a probe
format, not a model format contract.

Example requests:

```bash
curl http://127.0.0.1:2048/health
curl http://127.0.0.1:2048/v1/models
curl -H 'Content-Type: application/json' \
  -d '{"values":[1,2,3,4,5,6,7,8]}' \
  http://127.0.0.1:2048/v1/infer
curl -N -H 'Content-Type: application/json' \
  -d '{"values":[1,2,3,4,5,6,7,8]}' \
  http://127.0.0.1:2048/v1/stream
curl http://127.0.0.1:2048/metrics
```

`--max-connections=N` makes an automated run stop after N accepted
connections. Without it, Ctrl-C/SIGTERM enters the cancellation and scheduler
shutdown path.

## What this probe intentionally does not pretend

The fixed-schema parser is sample-only. A production engine should use a
mature JSON/schema dependency. Model format, tokenization, batching, admission,
authentication, routing policy, and inference semantics belong to the engine.
The [consumer-portfolio audit](../../docs/implementation/tasks/project/public-api-consumer-portfolio-audit-2026-09.md)
records the generic glue that this slice still has to implement itself. Those
findings are capability candidates or named product triggers, not a silent claim
that the current API is already a complete server framework.
