"""Re-export LumenRL's ``MooncakeConfig`` under the ``torchspec`` path.

ATOM calls ``MooncakeConfig(**mooncake_config)`` with the keys
``local_hostname``, ``metadata_server``, ``master_server_address``,
``protocol``, ``device_name``, ``global_segment_size``, ``local_buffer_size``,
``enable_gpu_direct``, ``enable_hard_pin``, ``max_seq_len``, ``hidden_dim`` and
``host_buffer_size`` — all of which are fields on the LumenRL dataclass.
"""

from lumenrl.transfer.mooncake_config import MooncakeConfig

__all__ = ["MooncakeConfig"]
