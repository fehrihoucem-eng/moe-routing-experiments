import sys; sys.path.insert(0, "src")
from routetrace.capture import capture

replies = capture(
    prompts=["What is 2+2?", "Name one primary colour.", "Say hi."],
    trace_path="data/traces/serve3.trace",
    model_dir="/home/houcem-fehri/Models/qwen36_i4_gs64",
    engine="/home/houcem-fehri/colibri/c/qwen36",
    max_tok=16,
    env_extra={"COLI_CUDA": "1", "COLI_GPUS": "0", "CUDA_EXPERT_GB": "auto",
               "HEAT_FILE": "/home/houcem-fehri/colibri-run/heat.bin",
               "OMP_NUM_THREADS": "16"},
)
for r in replies:
    print(f"[{r.key}] np={r.n_prompt_tokens} err={r.error} text={r.text[:60]!r}")
