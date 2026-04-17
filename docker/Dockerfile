FROM rocm/pytorch:latest

WORKDIR /workspace

RUN pip install --no-cache-dir \
    ray[default]>=2.9 \
    omegaconf>=2.3 \
    transformers>=4.40 \
    datasets>=2.18 \
    accelerate>=0.28

COPY . /workspace/Lumen-RL/

RUN cd /workspace/Lumen-RL && pip install -e ".[all]"

WORKDIR /workspace/Lumen-RL

ENTRYPOINT ["python"]
CMD ["examples/run_grpo.py", "--config", "configs/grpo_dense_bf16.yaml"]
