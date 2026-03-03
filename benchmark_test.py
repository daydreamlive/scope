import os
import shutil
import struct
import time

import onnx.helper

if not hasattr(onnx.helper, "float32_to_bfloat16"):
    def _float32_to_bfloat16(v):
        return struct.unpack("H", struct.pack("f", v)[2:4])[0]
    onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
import tensorrt as trt
import torch
from onnx import shape_inference
from onnx.external_data_helper import convert_model_to_external_data, write_external_data_tensors

from scope.core.pipelines.longlive.modules.causal_model import CausalWanModel
from scope.core.pipelines.wan2_1.components.generator import WanDiffusionWrapper

# --- Config ---
MODEL_DIR = os.path.expanduser("~/.daydream-scope/models")
GENERATOR_PATH = os.path.join(MODEL_DIR, "LongLive-1.3B/models/longlive_base.pt")
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
WARMUP_ITERS = 3
BENCH_ITERS = 20
ONNX_DIR = "onnx_models/benchmark"

# with-mask paths
ONNX_RAW        = os.path.join(ONNX_DIR, "model.onnx")
ONNX_OPT        = os.path.join(ONNX_DIR, "model_opt.onnx")
TRT_ENGINE_PATH = os.path.join(ONNX_DIR, "model_opt.engine")

# no-mask paths
ONNX_RAW_NM        = os.path.join(ONNX_DIR, "model_nomask.onnx")
ONNX_OPT_NM        = os.path.join(ONNX_DIR, "model_opt_nomask.onnx")
TRT_ENGINE_NM_PATH = os.path.join(ONNX_DIR, "model_opt_nomask.engine")

# Delete all previously compiled artifacts so everything is rebuilt fresh
if os.path.exists(ONNX_DIR):
    shutil.rmtree(ONNX_DIR)
    print(f"[CLEANUP] Removed {ONNX_DIR}")

GiB = 2 ** 30

_TORCH_TO_NP_DTYPE = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.uint16,
    torch.int64: np.int64,
    torch.int32: np.int32,
}

_TRT_DTYPE_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.bfloat16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
}


def bench(name, fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    median = sorted(times)[len(times) // 2]
    print(f"\n{name} ({iters} iters):")
    print(f"  Average: {avg * 1000:.2f} ms")
    print(f"  Median:  {median * 1000:.2f} ms")
    print(f"  Best:    {min(times) * 1000:.2f} ms")
    print(f"  Worst:   {max(times) * 1000:.2f} ms")
    return times


# =====================================================================
# ONNX graph optimizer (graphsurgeon)
# =====================================================================

class Optimizer:
    def __init__(self, onnx_graph, verbose=True):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"  {prefix}: {len(self.graph.nodes)} nodes, "
                f"{len(self.graph.tensors().keys())} tensors, "
                f"{len(self.graph.inputs)} inputs, "
                f"{len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        self.graph.fold_constants()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def infer_shapes(self, return_onnx=False):
        tmp_in = os.path.join(ONNX_DIR, "_tmp_shape_in.onnx")
        tmp_out = os.path.join(ONNX_DIR, "_tmp_shape_out.onnx")
        data_file = "_tmp_shape_weights.pb"
        data_path = os.path.join(ONNX_DIR, data_file)

        onnx_graph = gs.export_onnx(self.graph)
        onnx.save_model(
            onnx_graph, tmp_in,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=1024,
        )
        shape_inference.infer_shapes_path(tmp_in, tmp_out)
        onnx_graph = onnx.load(tmp_out, load_external_data=True)
        self.graph = gs.import_onnx(onnx_graph)

        for f in [tmp_in, tmp_out, data_path]:
            if os.path.exists(f):
                os.remove(f)
        if return_onnx:
            return gs.export_onnx(self.graph)

    def optimize(self):
        self.info("original")
        self.cleanup()
        self.info("cleanup")
        self.fold_constants()
        self.info("fold constants")
        self.cleanup()
        self.info("fold cleanup")
        self.infer_shapes()
        self.info("shape inference")
        self.cleanup()
        self.info("final")


def save_large_onnx(model_proto, path, data_file):
    """Save an ONNX model that may exceed 2 GB via external data.

    gs.export_onnx may store some tensors as float_data/int32_data instead
    of raw_data. convert_model_to_external_data only externalises raw_data
    tensors, so unconverted ones stay inline and can push the proto past
    the 2 GB protobuf limit, silently corrupting them during serialization.
    Fix: coerce everything to raw_data first, then externalise with
    size_threshold=0 so the proto is guaranteed small.
    """
    from onnx import numpy_helper

    data_path = os.path.join(os.path.dirname(path), data_file)
    if os.path.exists(data_path):
        os.remove(data_path)

    for tensor in model_proto.graph.initializer:
        if tensor.raw_data:
            continue
        arr = numpy_helper.to_array(tensor)
        tensor.raw_data = arr.tobytes()
        for field in ("float_data", "int32_data", "int64_data",
                      "double_data", "uint64_data"):
            tensor.ClearField(field)

    convert_model_to_external_data(
        model_proto,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=1024,
        convert_attribute=False,
    )
    model_dir = os.path.dirname(path)
    write_external_data_tensors(model_proto, model_dir)
    with open(path, "wb") as f:
        f.write(model_proto.SerializeToString())


# =====================================================================
# TensorRT engine builder
# =====================================================================

class TRTEngine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = {}
        self.stream = torch.cuda.Stream()

    def build(self, onnx_path, fp16=True, workspace_size=None):
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        print(f"  Parsing ONNX from {onnx_path}...")
        success = parser.parse_from_file(onnx_path)
        if not success:
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")
        print(f"  Network: {network.num_inputs} inputs, {network.num_outputs} outputs, "
              f"{network.num_layers} layers")

        config = builder.create_builder_config()
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        free_mem, total_mem = torch.cuda.mem_get_info()
        if workspace_size is None:
            if free_mem > 6 * GiB:
                workspace_size = free_mem - 4 * GiB
            else:
                workspace_size = max(free_mem - 1 * GiB, 1 * GiB)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        print(f"  Workspace: {workspace_size / GiB:.1f} GiB "
              f"(GPU free: {free_mem / GiB:.1f} / {total_mem / GiB:.1f} GiB)")

        print("  Building engine (this may take several minutes)...")
        t0 = time.time()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        engine_bytes = serialized.nbytes if hasattr(serialized, "nbytes") else len(serialized)
        print(f"  Engine built in {time.time() - t0:.1f}s "
              f"({engine_bytes / 2**20:.0f} MiB)")

        engine_data = bytes(serialized)
        with open(self.engine_path, "wb") as f:
            f.write(engine_data)
        print(f"  Saved to {self.engine_path}")

        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)

    def load(self):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        print(f"  Loaded engine from {self.engine_path}")

    def activate(self, inputs):
        """Create execution context and allocate output buffers."""
        self.context = self.engine.create_execution_context()

        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.tensors[name] = tensor

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = _TRT_DTYPE_TO_TORCH[self.engine.get_tensor_dtype(name)]
                self.tensors[name] = torch.empty(shape, dtype=dtype, device="cuda")
                print(f"  Output '{name}': {shape} {dtype}")

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

    def infer(self):
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()


# =====================================================================
# Helpers
# =====================================================================

def op_summary(onnx_path):
    m = onnx.load(onnx_path, load_external_data=False)
    counts = {}
    for node in m.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    print(f"  Op counts (top 15):")
    for op, c in sorted(counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {op}: {c}")
    print(f"  Total nodes: {sum(counts.values())}")


def make_ort_session(onnx_path):
    opts = ort.SessionOptions()
    # opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 8
    return ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=[("CUDAExecutionProvider", {
            "device_id": 0,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        })],
    )


def capture_ort_output(session, input_list):
    io = session.io_binding()
    for name, tensor in input_list:
        io.bind_input(name, "cuda", 0, _TORCH_TO_NP_DTYPE[tensor.dtype], tuple(tensor.shape), tensor.data_ptr())
    io.bind_output("output", "cuda", 0)
    session.run_with_iobinding(io)
    return torch.as_tensor(io.get_outputs()[0].numpy(), device="cuda").float()


def optimize_onnx(raw_path, opt_path, weights_file):
    t0 = time.time()
    raw_model = onnx.load(raw_path, load_external_data=True)
    opt = Optimizer(raw_model)
    del raw_model
    opt.optimize()
    opt_graph = gs.export_onnx(opt.graph)
    save_large_onnx(opt_graph, opt_path, weights_file)
    del opt, opt_graph
    try:
        _ = onnx.load(opt_path, load_external_data=True)
        del _
        print("  [verified] optimized model loads OK")
    except Exception as e:
        print(f"  [WARNING] optimized model failed to reload: {e}")
    print(f"  Optimization done in {time.time() - t0:.2f}s")


# =====================================================================
# Load model
# =====================================================================
print("Loading CausalWanModel via WanDiffusionWrapper...")
t0 = time.time()
wrapper = WanDiffusionWrapper(
    CausalWanModel,
    model_name="Wan2.1-T2V-1.3B",
    model_dir=MODEL_DIR,
    generator_path=GENERATOR_PATH,
    generator_model_name="generator",
)
model = wrapper.model.to(device=DEVICE, dtype=DTYPE)
model.eval()
print(f"Model loaded in {time.time() - t0:.2f}s\n")

# --- Build dummy inputs ---
x = torch.randn(1, 16, 3, 60, 104, device=DEVICE, dtype=DTYPE)
t_steps = torch.randint(0, 1000, (1, 3), device=DEVICE, dtype=torch.int64)
context = torch.randn(1, 512, 4096, device=DEVICE, dtype=DTYPE)
freqs_cos = torch.randn(1, 4680, 1, 64, device=DEVICE, dtype=DTYPE)
freqs_sin = torch.randn(1, 4680, 1, 64, device=DEVICE, dtype=DTYPE)
cache_ks = torch.zeros(30, 1, 14040, 12, 128, device=DEVICE, dtype=DTYPE)
cache_vs = torch.zeros(30, 1, 14040, 12, 128, device=DEVICE, dtype=DTYPE)

seq_len = 4680
cache_size = 14040
mask = torch.full((1, 1, 1, cache_size + seq_len), float("-inf"), device=DEVICE, dtype=DTYPE)
mask[:, :, :, cache_size:] = 0.0

results = {}
outputs = {}
memory = {}

# =====================================================================
# 1) PyTorch benchmark — with mask and without mask
# =====================================================================
print("=" * 60)
print("1) PyTorch bf16 — with mask")
print("=" * 60)
torch.cuda.reset_peak_memory_stats()
times = bench(
    "PyTorch bf16 with mask",
    lambda: model(x, t_steps, context, freqs_cos, freqs_sin, cache_ks, cache_vs, mask),
)
results["pytorch_mask"] = sum(times) / len(times) * 1000
memory["pytorch_mask"] = torch.cuda.max_memory_allocated() / GiB

print("\n" + "=" * 60)
print("2) PyTorch bf16 — no mask")
print("=" * 60)
torch.cuda.reset_peak_memory_stats()
times = bench(
    "PyTorch bf16 no mask",
    lambda: model(x, t_steps, context, freqs_cos, freqs_sin, cache_ks, cache_vs, None),
)
results["pytorch_nomask"] = sum(times) / len(times) * 1000
memory["pytorch_nomask"] = torch.cuda.max_memory_allocated() / GiB

# capture reference outputs before the model is deleted
with torch.no_grad():
    outputs["pytorch_mask"]   = model(x, t_steps, context, freqs_cos, freqs_sin, cache_ks, cache_vs, mask)[0].float()
    outputs["pytorch_nomask"] = model(x, t_steps, context, freqs_cos, freqs_sin, cache_ks, cache_vs, None)[0].float()

# =====================================================================
# 2) ONNX export — with mask
# =====================================================================
os.makedirs(ONNX_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("3) ONNX export — with mask (fp16)")
print("=" * 60)
model.half()
dummy_mask = (
    x.half(), t_steps, context.half(), freqs_cos.half(), freqs_sin.half(),
    cache_ks.half(), cache_vs.half(), mask.half(),
)
print("  Exporting...")
t0 = time.time()
torch.onnx.export(
    model, dummy_mask, ONNX_RAW,
    export_params=True, opset_version=17, do_constant_folding=True,
    input_names=["x", "t", "context", "freqs_cos", "freqs_sin", "cache_ks", "cache_vs", "mask"],
    output_names=["output", "new_ks", "new_vs"],
    dynamo=False,
)
raw_proto = onnx.load(ONNX_RAW)
save_large_onnx(raw_proto, ONNX_RAW, "weights.pb")
del raw_proto
print(f"  ONNX export (with mask) done in {time.time() - t0:.2f}s")

# =====================================================================
# 3) ONNX export — no mask
# =====================================================================
print("\n" + "=" * 60)
print("4) ONNX export — no mask (fp16)")
print("=" * 60)

# Wrap the model so mask=None is baked into the trace (no mask input in graph)
class _ModelNoMask(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x, t, context, freqs_cos, freqs_sin, cache_ks, cache_vs):
        return self.m(x, t, context, freqs_cos, freqs_sin, cache_ks, cache_vs, None)

model_nm = _ModelNoMask(model)
dummy_nomask = (
    x.half(), t_steps, context.half(), freqs_cos.half(), freqs_sin.half(),
    cache_ks.half(), cache_vs.half(),
)
print("  Exporting...")
t0 = time.time()
torch.onnx.export(
    model_nm, dummy_nomask, ONNX_RAW_NM,
    export_params=True, opset_version=17, do_constant_folding=True,
    input_names=["x", "t", "context", "freqs_cos", "freqs_sin", "cache_ks", "cache_vs"],
    output_names=["output", "new_ks", "new_vs"],
    dynamo=False,
)
raw_proto_nm = onnx.load(ONNX_RAW_NM)
save_large_onnx(raw_proto_nm, ONNX_RAW_NM, "weights_nomask.pb")
del raw_proto_nm, model_nm
print(f"  ONNX export (no mask) done in {time.time() - t0:.2f}s")

del model, wrapper
torch.cuda.empty_cache()

# =====================================================================
# 4) Optimize ONNX graphs
# =====================================================================
print("\n" + "=" * 60)
print("5) ONNX graph optimization — with mask")
print("=" * 60)
optimize_onnx(ONNX_RAW, ONNX_OPT, "weights_opt.pb")

print("\n" + "=" * 60)
print("6) ONNX graph optimization — no mask")
print("=" * 60)
optimize_onnx(ONNX_RAW_NM, ONNX_OPT_NM, "weights_opt_nomask.pb")

# =====================================================================
# 5) Op summary
# =====================================================================
print("\n" + "=" * 60)
print("Graph summary")
print("=" * 60)
print(f"\nWith mask — raw ({ONNX_RAW}):");    op_summary(ONNX_RAW)
print(f"\nWith mask — opt ({ONNX_OPT}):");    op_summary(ONNX_OPT)
print(f"\nNo mask — raw ({ONNX_RAW_NM}):");   op_summary(ONNX_RAW_NM)
print(f"\nNo mask — opt ({ONNX_OPT_NM}):");   op_summary(ONNX_OPT_NM)

# =====================================================================
# 6) ORT benchmarks — with mask
# =====================================================================
device_id = 0

fp16_inputs_mask = {
    "x":         x.half().contiguous(),
    "t":         t_steps.contiguous(),
    "context":   context.half().contiguous(),
    "freqs_cos": freqs_cos.half().contiguous(),
    "freqs_sin": freqs_sin.half().contiguous(),
    "cache_ks":  cache_ks.half().contiguous(),
    "cache_vs":  cache_vs.half().contiguous(),
    "mask":      mask.half().contiguous(),
}

fp16_inputs_nomask = {
    "x":         x.half().contiguous(),
    "t":         t_steps.contiguous(),
    "context":   context.half().contiguous(),
    "freqs_cos": freqs_cos.half().contiguous(),
    "freqs_sin": freqs_sin.half().contiguous(),
    "cache_ks":  cache_ks.half().contiguous(),
    "cache_vs":  cache_vs.half().contiguous(),
}

ort_list_mask   = list(fp16_inputs_mask.items())
ort_list_nomask = list(fp16_inputs_nomask.items())


def run_ort(session, input_list):
    io = session.io_binding()
    for name, tensor in input_list:
        io.bind_input(
            name, "cuda", device_id,
            _TORCH_TO_NP_DTYPE[tensor.dtype],
            tuple(tensor.shape), tensor.data_ptr(),
        )
    for name in ["output", "new_ks", "new_vs"]:
        io.bind_output(name, "cuda", device_id)
    session.run_with_iobinding(io)


print("\n" + "=" * 60)
print("7) ORT CUDA EP — raw ONNX with mask (fp16)")
print("=" * 60)
try:
    sess = make_ort_session(ONNX_RAW)
    print(f"  Providers: {sess.get_providers()}")
    torch.cuda.reset_peak_memory_stats()
    times = bench("ORT raw with mask", lambda: run_ort(sess, ort_list_mask))
    results["ort_raw_mask"] = sum(times) / len(times) * 1000
    memory["ort_raw_mask"] = torch.cuda.max_memory_allocated() / GiB
    outputs["ort_raw_mask"] = capture_ort_output(sess, ort_list_mask)
    del sess
except Exception as e:
    print(f"  [FAILED] {e}")
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("8) ORT CUDA EP — raw ONNX no mask (fp16)")
print("=" * 60)
try:
    sess = make_ort_session(ONNX_RAW_NM)
    print(f"  Providers: {sess.get_providers()}")
    torch.cuda.reset_peak_memory_stats()
    times = bench("ORT raw no mask", lambda: run_ort(sess, ort_list_nomask))
    results["ort_raw_nomask"] = sum(times) / len(times) * 1000
    memory["ort_raw_nomask"] = torch.cuda.max_memory_allocated() / GiB
    outputs["ort_raw_nomask"] = capture_ort_output(sess, ort_list_nomask)
    del sess
except Exception as e:
    print(f"  [FAILED] {e}")
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("9) ORT CUDA EP — optimized ONNX with mask (fp16)")
print("=" * 60)
try:
    sess = make_ort_session(ONNX_OPT)
    print(f"  Providers: {sess.get_providers()}")
    torch.cuda.reset_peak_memory_stats()
    times = bench("ORT opt with mask", lambda: run_ort(sess, ort_list_mask))
    results["ort_opt_mask"] = sum(times) / len(times) * 1000
    memory["ort_opt_mask"] = torch.cuda.max_memory_allocated() / GiB
    outputs["ort_opt_mask"] = capture_ort_output(sess, ort_list_mask)
    del sess
except Exception as e:
    print(f"  [FAILED] {e}")
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("10) ORT CUDA EP — optimized ONNX no mask (fp16)")
print("=" * 60)
try:
    sess = make_ort_session(ONNX_OPT_NM)
    print(f"  Providers: {sess.get_providers()}")
    torch.cuda.reset_peak_memory_stats()
    times = bench("ORT opt no mask", lambda: run_ort(sess, ort_list_nomask))
    results["ort_opt_nomask"] = sum(times) / len(times) * 1000
    memory["ort_opt_nomask"] = torch.cuda.max_memory_allocated() / GiB
    outputs["ort_opt_nomask"] = capture_ort_output(sess, ort_list_nomask)
    del sess
except Exception as e:
    print(f"  [FAILED] {e}")
torch.cuda.empty_cache()

# =====================================================================
# 7) TensorRT — with mask
# =====================================================================
print("\n" + "=" * 60)
print("11) TensorRT fp16 — with mask")
print("=" * 60)
try:
    trt_mask = TRTEngine(TRT_ENGINE_PATH)
    if os.path.exists(TRT_ENGINE_PATH):
        trt_mask.load()
    else:
        trt_mask.build(ONNX_OPT, fp16=True)
    trt_mask.activate(fp16_inputs_mask)
    torch.cuda.reset_peak_memory_stats()
    times = bench("TensorRT fp16 with mask", trt_mask.infer)
    results["trt_mask"] = sum(times) / len(times) * 1000
    memory["trt_mask"] = torch.cuda.max_memory_allocated() / GiB
    outputs["trt_mask"] = trt_mask.tensors["output"].float()
except Exception as e:
    print(f"  [FAILED] {e}")
torch.cuda.empty_cache()

# =====================================================================
# 8) TensorRT — no mask
# =====================================================================
print("\n" + "=" * 60)
print("12) TensorRT fp16 — no mask")
print("=" * 60)
try:
    trt_nomask = TRTEngine(TRT_ENGINE_NM_PATH)
    if os.path.exists(TRT_ENGINE_NM_PATH):
        trt_nomask.load()
    else:
        trt_nomask.build(ONNX_OPT_NM, fp16=True)
    trt_nomask.activate(fp16_inputs_nomask)
    torch.cuda.reset_peak_memory_stats()
    times = bench("TensorRT fp16 no mask", trt_nomask.infer)
    results["trt_nomask"] = sum(times) / len(times) * 1000
    memory["trt_nomask"] = torch.cuda.max_memory_allocated() / GiB
    outputs["trt_nomask"] = trt_nomask.tensors["output"].float()
except Exception as e:
    print(f"  [FAILED] {e}")
torch.cuda.empty_cache()

# =====================================================================
# Output accuracy comparison
# =====================================================================
def _cmp(a, b):
    diff = (a - b).abs()
    return diff.max().item(), diff.mean().item()

print("\n" + "=" * 60)
print("OUTPUT ACCURACY — WITH MASK  (vs PyTorch reference)")
print("=" * 60)
print(f"  {'Backend':<22}  {'max|diff|':>12}  {'mean|diff|':>12}")
print(f"  {'-'*22}  {'-'*12}  {'-'*12}")
ref = outputs.get("pytorch_mask")
for label, key in [("ORT raw fp16", "ort_raw_mask"), ("ORT opt fp16", "ort_opt_mask"), ("TRT fp16", "trt_mask")]:
    out = outputs.get(key)
    if ref is None or out is None:
        print(f"  {label:<22}  {'N/A':>12}  {'N/A':>12}")
    else:
        mx, mn = _cmp(out, ref)
        print(f"  {label:<22}  {mx:>12.6f}  {mn:>12.6f}")

print("\n" + "=" * 60)
print("OUTPUT ACCURACY — NO MASK  (vs PyTorch reference)")
print("=" * 60)
print(f"  {'Backend':<22}  {'max|diff|':>12}  {'mean|diff|':>12}")
print(f"  {'-'*22}  {'-'*12}  {'-'*12}")
ref_nm = outputs.get("pytorch_nomask")
for label, key in [("ORT raw fp16", "ort_raw_nomask"), ("ORT opt fp16", "ort_opt_nomask"), ("TRT fp16", "trt_nomask")]:
    out = outputs.get(key)
    if ref_nm is None or out is None:
        print(f"  {label:<22}  {'N/A':>12}  {'N/A':>12}")
    else:
        mx, mn = _cmp(out, ref_nm)
        print(f"  {label:<22}  {mx:>12.6f}  {mn:>12.6f}")

# =====================================================================
# Final comparison table
# =====================================================================
print("\n" + "=" * 60)
print("FINAL COMPARISON  (avg latency, lower is better)")
print("=" * 60)

rows = [
    ("PyTorch bf16",      "pytorch_mask",    "pytorch_nomask"),
    ("ORT raw fp16",      "ort_raw_mask",    "ort_raw_nomask"),
    ("ORT opt fp16",      "ort_opt_mask",    "ort_opt_nomask"),
    ("TensorRT fp16",     "trt_mask",        "trt_nomask"),
]
print(f"  {'Backend':<22}  {'with mask':>12}  {'VRAM':>8}  {'no mask':>12}  {'VRAM':>8}  {'diff':>12}  {'overhead':>10}")
print(f"  {'-'*22}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*10}")
for label, k_mask, k_nm in rows:
    m   = results.get(k_mask)
    nm  = results.get(k_nm)
    vm  = memory.get(k_mask)
    vnm = memory.get(k_nm)
    if m is not None and nm is not None:
        diff = m - nm
        pct  = diff / nm * 100
        vm_str  = f"{vm:.1f}GB" if vm is not None else "N/A"
        vnm_str = f"{vnm:.1f}GB" if vnm is not None else "N/A"
        print(f"  {label:<22}  {m:>11.2f}ms  {vm_str:>8}  {nm:>11.2f}ms  {vnm_str:>8}  {diff:>+11.2f}ms  {pct:>+9.1f}%")
    elif m is not None:
        vm_str = f"{vm:.1f}GB" if vm is not None else "N/A"
        print(f"  {label:<22}  {m:>11.2f}ms  {vm_str:>8}  {'N/A':>12}  {'N/A':>8}  {'N/A':>12}  {'N/A':>10}")
    elif nm is not None:
        vnm_str = f"{vnm:.1f}GB" if vnm is not None else "N/A"
        print(f"  {label:<22}  {'N/A':>12}  {'N/A':>8}  {nm:>11.2f}ms  {vnm_str:>8}  {'N/A':>12}  {'N/A':>10}")
    else:
        print(f"  {label:<22}  {'N/A':>12}  {'N/A':>8}  {'N/A':>12}  {'N/A':>8}  {'N/A':>12}  {'N/A':>10}")
