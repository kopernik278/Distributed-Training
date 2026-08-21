from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .model import compute_language_model_loss
from .parallel_state import (
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from .profiler import record_range


@dataclass
class MicrobatchIO:
    input_ids: torch.Tensor
    targets: torch.Tensor


@dataclass
class _ForwardBundle:
    """Tensors retained so backward can run after later microbatches."""

    output: torch.Tensor
    input_activation: torch.Tensor | None  # received activation (non-first stages)
    loss: torch.Tensor | None  # only on last stage


def _wait(req: dist.Work | None) -> None:
    if req is not None:
        req.wait()


class PipelineEngine:
    """Educational 1F1B pipeline engine with non-blocking P2P.

    Blocking send/recv deadlocks in the steady phase (rank0 wants to send the next
    activation while rank1 wants to send the previous gradient). ``isend``/``irecv``
    lets both directions progress.

    ``stage`` may be a raw ``PipelineStage`` or a DDP-wrapped module (required when
    DP>1 so gradient hooks still fire).
    """

    def __init__(
        self,
        stage: torch.nn.Module,
        *,
        num_microbatches: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        sequence_parallel: bool = False,
        vocab_parallel: bool = False,
    ) -> None:
        if num_microbatches < 1:
            raise ValueError("num_microbatches must be >= 1")
        self.stage = stage
        self.num_microbatches = num_microbatches
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.sequence_parallel = sequence_parallel
        self.vocab_parallel = vocab_parallel
        self.pp_size = get_pipeline_model_parallel_world_size()
        self.pp_rank = get_pipeline_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _activation_seq_len(self, full_seq_len: int) -> int:
        if self.sequence_parallel and self.tp_size > 1:
            return full_seq_len // self.tp_size
        return full_seq_len

    def _recv_forward(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        prev = get_pipeline_model_parallel_prev_rank()
        assert prev is not None
        buf = torch.empty(batch_size, seq_len, self.hidden_size, dtype=self.dtype, device=device)
        with record_range("pp_recv_forward"):
            dist.recv(buf, src=prev)
        return buf

    def _send_forward_async(self, hidden: torch.Tensor) -> dist.Work:
        nxt = get_pipeline_model_parallel_next_rank()
        assert nxt is not None
        with record_range("pp_isend_forward"):
            return dist.isend(hidden.contiguous(), dst=nxt)

    def _recv_backward_async(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, dist.Work]:
        nxt = get_pipeline_model_parallel_next_rank()
        assert nxt is not None
        buf = torch.empty(batch_size, seq_len, self.hidden_size, dtype=self.dtype, device=device)
        with record_range("pp_irecv_backward"):
            work = dist.irecv(buf, src=nxt)
        return buf, work

    def _send_backward_async(self, grad: torch.Tensor) -> dist.Work:
        prev = get_pipeline_model_parallel_prev_rank()
        assert prev is not None
        with record_range("pp_isend_backward"):
            return dist.isend(grad.contiguous(), dst=prev)

    def _forward_microbatch(
        self, mb: MicrobatchIO, device: torch.device
    ) -> tuple[_ForwardBundle, dist.Work | None]:
        input_activation: torch.Tensor | None = None
        if is_pipeline_first_stage():
            output = self.stage(input_ids=mb.input_ids)
        else:
            batch_size, full_seq = mb.input_ids.shape
            act_seq = self._activation_seq_len(full_seq)
            hidden = self._recv_forward(batch_size, act_seq, device)
            hidden = hidden.detach().requires_grad_(True)
            input_activation = hidden
            output = self.stage(hidden_states=hidden)

        loss: torch.Tensor | None = None
        send_req: dist.Work | None = None
        if is_pipeline_last_stage():
            loss = (
                compute_language_model_loss(
                    output, mb.targets, vocab_parallel=self.vocab_parallel
                )
                / self.num_microbatches
            )
        else:
            send_req = self._send_forward_async(output.detach())

        return _ForwardBundle(output=output, input_activation=input_activation, loss=loss), send_req

    def _backward_microbatch(
        self,
        bundle: _ForwardBundle,
        *,
        pending_forward_send: dist.Work | None,
    ) -> dist.Work | None:
        """Returns optional pending send_backward work."""
        if is_pipeline_last_stage():
            assert bundle.loss is not None
            bundle.loss.backward()
            if bundle.input_activation is not None and bundle.input_activation.grad is not None:
                return self._send_backward_async(bundle.input_activation.grad)
            return None

        batch_size, seq_len, _ = bundle.output.shape
        grad_buf, recv_req = self._recv_backward_async(batch_size, seq_len, bundle.output.device)
        # Overlap: wait for prior forward send together with incoming grad.
        _wait(pending_forward_send)
        recv_req.wait()
        torch.autograd.backward(bundle.output, grad_tensors=grad_buf)
        if bundle.input_activation is not None and bundle.input_activation.grad is not None:
            return self._send_backward_async(bundle.input_activation.grad)
        return None

    def run(self, microbatches: list[MicrobatchIO], device: torch.device) -> float:
        """Run one optimizer-step worth of 1F1B. Returns summed microbatch loss on last stage."""
        if len(microbatches) != self.num_microbatches:
            raise ValueError(
                f"expected {self.num_microbatches} microbatches, got {len(microbatches)}"
            )

        num_warmup = min(self.pp_size - self.pp_rank - 1, self.num_microbatches)
        num_steady = self.num_microbatches - num_warmup

        fwd_idx = 0
        in_flight: list[tuple[_ForwardBundle, dist.Work | None]] = []
        pending_bwd_send: dist.Work | None = None
        loss_sum = 0.0

        for _ in range(num_warmup):
            bundle, send_req = self._forward_microbatch(microbatches[fwd_idx], device)
            in_flight.append((bundle, send_req))
            fwd_idx += 1

        for _ in range(num_steady):
            bundle, send_req = self._forward_microbatch(microbatches[fwd_idx], device)
            in_flight.append((bundle, send_req))
            fwd_idx += 1

            done, done_send = in_flight.pop(0)
            # Previous backward send must complete before we reuse the peer's recv slot.
            _wait(pending_bwd_send)
            pending_bwd_send = self._backward_microbatch(done, pending_forward_send=done_send)
            if done.loss is not None:
                loss_sum += float(done.loss.detach().item())

        while in_flight:
            done, done_send = in_flight.pop(0)
            _wait(pending_bwd_send)
            pending_bwd_send = self._backward_microbatch(done, pending_forward_send=done_send)
            if done.loss is not None:
                loss_sum += float(done.loss.detach().item())

        _wait(pending_bwd_send)
        assert fwd_idx == self.num_microbatches
        return loss_sum
