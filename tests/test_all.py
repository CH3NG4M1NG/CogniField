"""
tests/test_all.py
=================
CogniField test suite — covers all 9 modules.
Run with:  PYTHONPATH=.. python -m pytest test_all.py -v
       or: PYTHONPATH=.. python test_all.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

PASS = 0; FAIL = 0; ERRORS = []

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ {name}" + (f" — {msg}" if msg else ""))
        FAIL += 1
        ERRORS.append(name)

# ─────────────────────────────────────────
# FrequencySpace
# ─────────────────────────────────────────
print("\n[FrequencySpace]")
from cognifield.latent_space.frequency_space import FrequencySpace, ComposeMode

space = FrequencySpace(dim=64)
rng   = np.random.default_rng(0)
a = space.l2(rng.standard_normal(64).astype(np.float32))
b = space.l2(rng.standard_normal(64).astype(np.float32))

check("l2_norm_unit", abs(np.linalg.norm(a) - 1.0) < 1e-5)
check("similarity_self",    abs(space.similarity(a, a) - 1.0) < 1e-5)
check("similarity_range",   -1.01 <= space.similarity(a, b) <= 1.01)
check("distance_range",      0.0 <= space.distance(a, b) <= 1.0)
check("compose_shape",       space.compose([a, b]).shape == (64,))
check("compose_unit",        abs(np.linalg.norm(space.compose([a,b])) - 1.0) < 1e-4)
for mode in ComposeMode:
    r = space.compose([a, b], mode=mode)
    check(f"compose_{mode.value}_shape", r.shape == (64,))
check("analogy_shape",  space.analogy(a, b, a).shape == (64,))
check("batch_similarity_shape",
      space.batch_similarity(a, np.stack([a,b])).shape == (2,))
check("nearest_shape",   len(space.nearest_in_batch(a, np.stack([a,b,a]), k=2)) == 2)

# ─────────────────────────────────────────
# TextEncoder
# ─────────────────────────────────────────
print("\n[TextEncoder]")
from cognifield.encoder.text_encoder import TextEncoder

enc = TextEncoder(dim=64); enc.fit()
va  = enc.encode("apple fruit food")
vb  = enc.encode("car engine drive")

check("encode_shape",  va.shape == (64,))
check("encode_unit",   abs(np.linalg.norm(va) - 1.0) < 1e-5)
check("encode_repro",  np.allclose(enc.encode("apple"), enc.encode("apple")))
check("semantic_sim",  enc.similarity("apple fruit", "eat apple") >
                        enc.similarity("apple fruit", "car engine drive"))
check("batch_shape",   enc.encode_batch(["a","b","c"]).shape == (3, 64))

# ─────────────────────────────────────────
# ImageEncoder
# ─────────────────────────────────────────
print("\n[ImageEncoder]")
from cognifield.encoder.image_encoder import ImageEncoder

ienc = ImageEncoder(dim=64)
img_a = np.random.rand(64, 64).astype(np.float32)
img_b = np.zeros((64,64,3), dtype=np.float32); img_b[:,:,0] = 0.9

check("encode_shape",  ienc.encode(img_a).shape == (64,))
check("encode_unit",   abs(np.linalg.norm(ienc.encode(img_a)) - 1.0) < 1e-5)
check("encode_repro",  np.allclose(ienc.encode(img_a), ienc.encode(img_a)))
check("random_shape",  ienc.encode_random().shape == (64,))
check("batch_shape",   ienc.encode_batch([img_a, img_a]).shape == (2, 64))

# ─────────────────────────────────────────
# AudioEncoder
# ─────────────────────────────────────────
print("\n[AudioEncoder]")
from cognifield.encoder.audio_encoder import AudioEncoder

aenc = AudioEncoder(dim=64)
wave = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)).astype(np.float32)

check("encode_shape",  aenc.encode(wave).shape == (64,))
check("encode_unit",   abs(np.linalg.norm(aenc.encode(wave)) - 1.0) < 1e-5)
check("sine_shape",    aenc.encode_sine().shape == (64,))

# ─────────────────────────────────────────
# MemoryStore
# ─────────────────────────────────────────
print("\n[MemoryStore]")
from cognifield.memory.memory_store import MemoryStore

mem = MemoryStore(dim=64, similarity_threshold=0.99)
rng2 = np.random.default_rng(7)
vecs = [space.l2(rng2.standard_normal(64).astype(np.float32)) for _ in range(10)]
for i, v in enumerate(vecs):
    mem.store(v, label=f"w{i}", modality="text")

check("store_count",   len(mem) == 10)
results = mem.retrieve(vecs[0], k=3)
check("retrieve_count", len(results) > 0)
check("retrieve_top",   results[0][0] > 0.95)

mem2 = MemoryStore(dim=64, similarity_threshold=0.90)
v = space.l2(rng2.standard_normal(64).astype(np.float32))
mem2.store(v, "a"); mem2.store(v.copy(), "b")
check("dedup",         len(mem2) == 1)

centroids = mem.cluster(n_clusters=3)
check("cluster_shape", centroids.shape == (3, 64))

# ─────────────────────────────────────────
# ReasoningEngine
# ─────────────────────────────────────────
print("\n[ReasoningEngine]")
from cognifield.reasoning.reasoning_engine import ReasoningEngine, ErrorType

rmem = MemoryStore(dim=64)
for text in ["apple", "stone", "eat"]:
    rmem.store(enc.encode(text), label=text, modality="text",
               allow_duplicate=True)

engine = ReasoningEngine(space=space, memory=rmem,
                          max_retries=4, threshold=0.65)

iv = enc.encode("I eat apple")
tv = enc.encode("apple")
result = engine.reason(iv, tv)
check("reason_returns", result.solution_vec.shape == (64,))
check("score_range",    0.0 <= result.score <= 1.0)
check("error_type_valid", isinstance(result.error_type, ErrorType))

err_none = engine.detect_error(tv, tv, score=0.9)
err_mismatch = engine.detect_error(tv, tv, score=0.2)
check("detect_none",     err_none    == ErrorType.NONE)
check("detect_mismatch", err_mismatch == ErrorType.SEMANTIC_MISMATCH)

# ─────────────────────────────────────────
# StructureChecker
# ─────────────────────────────────────────
print("\n[StructureChecker]")
from cognifield.language.structure_checker import StructureChecker

checker = StructureChecker()
good = checker.check("I eat apple in the morning.")
bad  = checker.check("the the a a")

check("good_score_high", good.overall_score >= 0.8)
check("bad_score_lower", bad.overall_score < good.overall_score)
check("report_valid",    good.is_valid)
check("issues_detected", len(bad.issues) > 0)
check("score_method",    0 <= checker.score("test text") <= 1)

# ─────────────────────────────────────────
# CuriosityEngine
# ─────────────────────────────────────────
print("\n[CuriosityEngine]")
from cognifield.curiosity.curiosity_engine import CuriosityEngine

cmem = MemoryStore(dim=64)
cengine = CuriosityEngine(memory=cmem, novelty_threshold=0.3)

check("empty_memory_novelty_1.0", cengine.detect_novelty(vecs[0]) == 1.0)
cmem.store(vecs[0], "known", allow_duplicate=True)
check("known_low_novelty", cengine.detect_novelty(vecs[0]) < 0.2)

rng3 = np.random.default_rng(123)
novel_vec = space.l2(rng3.standard_normal(64).astype(np.float32))
report = cengine.trigger_exploration(novel_vec, raw_input="xqz abc")
check("exploration_report", report.novelty_score >= 0.0)
check("hypothesis_str",     isinstance(report.hypothesis, str))
check("n_explorations",     cengine.n_explorations == 1)

# ─────────────────────────────────────────
# LossSystem
# ─────────────────────────────────────────
print("\n[LossSystem]")
from cognifield.loss.loss_system import LossSystem, LossConfig

ls = LossSystem(config=LossConfig(), space=space)
pred = space.l2(rng.standard_normal(64).astype(np.float32))
tgt  = space.l2(rng.standard_normal(64).astype(np.float32))

rec = ls.compute(pred, tgt, novelty=0.5)
check("error_loss_range",  0.0 <= rec.error_loss <= 1.0)
check("total_loss_range",  0.0 <= rec.total_loss <= 2.0)
check("novel_flagged",     rec.is_novel)

# Same → error_loss should be ~0
rec2 = ls.compute(tgt, tgt)
check("zero_error",        rec2.error_loss < 0.01)

summ = ls.summary()
check("summary_n_steps",   summ["n_steps"] == 2)

# ─────────────────────────────────────────
# SimpleEnv
# ─────────────────────────────────────────
print("\n[SimpleEnv]")
from cognifield.environment.simple_env import SimpleEnv

env = SimpleEnv(seed=0)
check("objects_loaded",  len(env.object_names) >= 6)
fb = env.step("observe")
check("observe_success", fb["success"])
check("state_vec_shape", fb["state_vec"].shape == (64,))

fb2 = env.step("move", 3, 3)
check("move_success",    fb2["success"])
check("move_pos",        env._agent_pos == (3, 3))

fb3 = env.step("eat", "lamp")  # lamp not edible
check("eat_inedible",    not fb3["success"])

# ─────────────────────────────────────────
# Agent (smoke test)
# ─────────────────────────────────────────
print("\n[CogniFieldAgent]")
from cognifield.agent.agent import CogniFieldAgent, AgentConfig

agent = CogniFieldAgent(config=AgentConfig(dim=64, seed=0),
                         env=SimpleEnv(seed=0))
s = agent.step("apple fruit food", modality="text", label="apple")
check("step_returns",    s.step == 1)
check("step_score",      0 <= s.reasoning_score <= 1.0)
check("memory_grew",     len(agent.memory) >= 1)

s2 = agent.step("I eat apple", modality="text", label="eat",
                action="observe")
check("env_reward_not_none", s2.env_reward is not None)

summ = agent.summary()
check("summary_steps",   summ["steps"] == 2)
check("summary_memory",  summ["memory_size"] >= 2)

# ─────────────────────────────────────────
print(f"\n{'═'*48}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All tests passed ✓")
print(f"{'═'*48}\n")
