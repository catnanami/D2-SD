import torch
import flashinfer


class CascadeGraphRunner:
    """
    CUDA graph runner for the local attention + merge portion of cascade attention.

    Shared attention (variable kv_len) runs as a regular kernel.
    Local attention + merge (fixed shapes) are captured in a CUDA graph per bsz,
    reducing kernel launch overhead from 2*N to 1 per layer.
    """

    def __init__(
        self,
        block_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        sm_scale: float,
        causal: bool,
    ):
        self.block_size = block_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.sm_scale = sm_scale
        self.causal = causal

        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._buffers: dict[int, dict[str, torch.Tensor]] = {}

    def _capture(self, bsz: int):
        """Pre-allocate buffers, warmup, and capture CUDA graph for given bsz."""
        bufs = {
            "q": torch.empty(bsz, self.block_size, self.num_heads, self.head_dim,
                             dtype=self.dtype, device=self.device),
            "k": torch.empty(bsz, self.block_size, self.num_kv_heads, self.head_dim,
                             dtype=self.dtype, device=self.device),
            "v": torch.empty(bsz, self.block_size, self.num_kv_heads, self.head_dim,
                             dtype=self.dtype, device=self.device),
            "out_shared": torch.empty(bsz, self.block_size, self.num_heads, self.head_dim,
                                      dtype=self.dtype, device=self.device),
            "lse_shared": torch.empty(bsz, self.block_size, self.num_heads,
                                      dtype=torch.float32, device=self.device),
        }

        def _run_local_merge():
            for i in range(bsz):
                out_l, lse_l = flashinfer.single_prefill_with_kv_cache(
                    bufs["q"][i], bufs["k"][i], bufs["v"][i],
                    causal=self.causal, return_lse=True, sm_scale=self.sm_scale,
                )
                flashinfer.merge_state_in_place(
                    bufs["out_shared"][i], bufs["lse_shared"][i], out_l, lse_l,
                )

        # Warmup
        for _ in range(3):
            _run_local_merge()
        torch.cuda.synchronize()

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _run_local_merge()
        torch.cuda.synchronize()

        self._graphs[bsz] = g
        self._buffers[bsz] = bufs

    def run(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        out_shared: torch.Tensor,
        lse_shared: torch.Tensor,
    ):
        """
        Run local attention + merge via CUDA graph replay.

        Args:
            q_local:    [bsz, block_size, num_heads, head_dim]
            k_local:    [bsz, block_size, num_kv_heads, head_dim]
            v_local:    [bsz, block_size, num_kv_heads, head_dim]
            out_shared: [bsz, block_size, num_heads, head_dim]  (shared attn output)
            lse_shared: [bsz, block_size, num_heads]             (shared attn lse)

        After return, out_shared contains the merged (shared + local) result.
        """
        bsz = q_local.shape[0]
        if bsz not in self._graphs:
            self._capture(bsz)

        bufs = self._buffers[bsz]
        bufs["q"].copy_(q_local)
        bufs["k"].copy_(k_local)
        bufs["v"].copy_(v_local)
        bufs["out_shared"].copy_(out_shared)
        bufs["lse_shared"].copy_(lse_shared)

        self._graphs[bsz].replay()

        out_shared.copy_(bufs["out_shared"])
        # Also surface the merged LSE so callers can apply post-merge corrections
        # (e.g. GPT-OSS attention-sink rescaling needs lse_combined).
        lse_shared.copy_(bufs["lse_shared"])
