# Eye tracker UART client

Adapted from `eye_tracker/src/macaque_tracker/{protocol,controller,uart}.py`
at commit `5c0e12e2e373f0ac0ac9bb4c723e7a42d93d7c39` (MIT; see LICENSE).
The wire format is specified in that project's `docs/protocol.md`.

Only controller-side encoding/decoding, request acknowledgement/retry logic,
and exclusive raw UART access are retained. Camera-dependent models, status
encoding, the Unix socket backend, sample iterator and CLI are omitted.
The decoded eye record is a local dataclass to avoid a camera/NumPy dependency.
Runtime configuration and eye-0 validity/filtering live in `bin/eye_tracking.py`.
Update these routines together when the tracker protocol changes.
