"""Consistency-only invariant checks for accelerators without per-shape
ground truth (V4-5).

Per Principle 2 of the v4 plan, KPU/TPU-class accelerators that lack
out-of-band measurement get a different validation contract than
CPU/GPU: instead of asserting per-shape predicted-vs-measured bands,
we assert that the analyzer's predictions are *internally consistent*
and physically plausible.

Each invariant is a pure function returning a list of failure messages
(empty list = pass). The pytest wrapper at
``tests/validation_model_v4/test_kpu_invariants.py`` parametrizes
the invariants over the KPU mapper family (T64/T128/T256/T768).
"""
