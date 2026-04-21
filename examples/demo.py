"""
examples/demo.py
================
CogniField Full Demonstration

Sections:
  1. Frequency Space — similarity and composition
  2. Text Encoding — semantic geometry
  3. Image Encoding — visual frequency representations
  4. Multimodal Linking — text ↔ image
  5. Reasoning Loop — self-correction with retries
  6. Curiosity Engine — novelty detection and exploration
  7. Language Checker — grammar and semantic scoring
  8. Environment Interaction — pick, move, eat, combine
  9. Full Agent Loop — integrated perceive-reason-act cycle
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def sep(title: str) -> None:
    print(f"\n{'─'*56}")
    print(f"  {title}")
    print(f"{'─'*56}")


# ═══════════════════════════════════════════════════════
# 1. Frequency Space
# ═══════════════════════════════════════════════════════
def demo_frequency_space():
    sep("1. Frequency Space — Similarity & Composition")

    from cognifield.latent_space.frequency_space import FrequencySpace, ComposeMode

    space = FrequencySpace(dim=64)
    rng   = np.random.default_rng(7)

    # Three concept vectors
    food    = rng.standard_normal(64).astype(np.float32)
    fruit   = rng.standard_normal(64).astype(np.float32)
    vehicle = rng.standard_normal(64).astype(np.float32)

    # Make fruit semantically close to food
    fruit = space.l2(food * 0.8 + fruit * 0.2)
    food  = space.l2(food)
    vehicle = space.l2(vehicle)

    print(f"\n  sim(food, fruit)   = {space.similarity(food, fruit):.4f}  (should be high)")
    print(f"  sim(food, vehicle) = {space.similarity(food, vehicle):.4f}  (should be low)")
    print(f"  dist(food, fruit)  = {space.distance(food, fruit):.4f}")

    # Composition
    composed = space.combine(food, fruit, mode=ComposeMode.MEAN)
    print(f"\n  composed(food+fruit) → sim to food  = {space.similarity(composed, food):.4f}")
    print(f"  composed(food+fruit) → sim to vehicle = {space.similarity(composed, vehicle):.4f}")

    # Analogy
    apple  = space.l2(food + rng.standard_normal(64).astype(np.float32) * 0.3)
    orange = space.l2(food + rng.standard_normal(64).astype(np.float32) * 0.3)
    query  = space.l2(rng.standard_normal(64).astype(np.float32))
    answer = space.analogy(apple, orange, query)
    print(f"\n  Analogy apple:orange :: query:?")
    print(f"    answer sim to food family: {space.similarity(answer, food):.4f}")


# ═══════════════════════════════════════════════════════
# 2. Text Encoding
# ═══════════════════════════════════════════════════════
def demo_text_encoding():
    sep("2. Text Encoding — Semantic Geometry")

    from cognifield.encoder.text_encoder import TextEncoder

    enc = TextEncoder(dim=64)
    enc.fit()

    pairs = [
        ("I eat apple",         "apple fruit food"),
        ("I eat apple",         "the car drives fast"),
        ("apple red fruit",     "orange yellow fruit"),
        ("pick up the stone",   "move the rock"),
        ("the agent learns",    "a cat meows"),
    ]

    print(f"\n  {'Text A':30s} | {'Text B':30s} | sim")
    print(f"  {'─'*30} | {'─'*30} | ─────")
    for a, b in pairs:
        sim = enc.similarity(a, b)
        bar = "█" * int(sim * 20) if sim > 0 else ""
        print(f"  {a[:30]:30s} | {b[:30]:30s} | {sim:.3f} {bar}")


# ═══════════════════════════════════════════════════════
# 3. Image Encoding
# ═══════════════════════════════════════════════════════
def demo_image_encoding():
    sep("3. Image Encoding — Visual Frequency Representation")

    from cognifield.encoder.image_encoder import ImageEncoder
    from cognifield.latent_space.frequency_space import FrequencySpace

    enc   = ImageEncoder(dim=64)
    space = FrequencySpace(dim=64)

    # Synthetic images with different structure
    rng = np.random.default_rng(42)

    def make_image(pattern: str) -> np.ndarray:
        img = np.zeros((64, 64), dtype=np.float32)
        if pattern == "solid_red":
            img[:] = 0.9
        elif pattern == "gradient":
            for i in range(64):
                img[i, :] = i / 64.0
        elif pattern == "checker":
            for i in range(64):
                for j in range(64):
                    img[i, j] = float((i // 8 + j // 8) % 2)
        elif pattern == "noisy":
            img = rng.random((64, 64)).astype(np.float32)
        elif pattern == "center_blob":
            cx, cy = 32, 32
            for i in range(64):
                for j in range(64):
                    img[i, j] = np.exp(-((i-cx)**2 + (j-cy)**2) / 200.0)
        return img

    patterns = ["solid_red", "gradient", "checker", "noisy", "center_blob"]
    vecs = {p: enc.encode(make_image(p)) for p in patterns}

    print("\n  Image similarity matrix:")
    print(f"  {'':14s}", end="")
    for p in patterns:
        print(f" {p[:9]:9s}", end="")
    print()
    for p1 in patterns:
        print(f"  {p1[:14]:14s}", end="")
        for p2 in patterns:
            sim = space.similarity(vecs[p1], vecs[p2])
            print(f"  {sim:+.3f} ", end="")
        print()


# ═══════════════════════════════════════════════════════
# 4. Multimodal Linking
# ═══════════════════════════════════════════════════════
def demo_multimodal_linking():
    sep("4. Multimodal Linking — Text ↔ Image")

    from cognifield.encoder.text_encoder  import TextEncoder
    from cognifield.encoder.image_encoder import ImageEncoder
    from cognifield.latent_space.frequency_space import FrequencySpace

    text_enc  = TextEncoder(dim=64);  text_enc.fit()
    image_enc = ImageEncoder(dim=64)
    space     = FrequencySpace(dim=64)

    # Encode text concepts
    t_apple = text_enc.encode("I eat apple red fruit")
    t_stone = text_enc.encode("pick up the heavy stone rock")
    t_water = text_enc.encode("pour water into glass liquid")

    # Synthetic images
    def apple_img():
        img = np.zeros((64,64,3), dtype=np.float32)
        img[:,:,0] = 0.9     # red channel dominant
        img[:,:,1] = 0.1
        img[:,:,2] = 0.1
        return img

    def stone_img():
        img = np.ones((64,64,3), dtype=np.float32) * 0.4
        return img

    def water_img():
        img = np.zeros((64,64,3), dtype=np.float32)
        img[:,:,2] = 0.8     # blue channel dominant
        return img

    i_apple = image_enc.encode(apple_img())
    i_stone = image_enc.encode(stone_img())
    i_water = image_enc.encode(water_img())

    print("\n  Cross-modal similarities (before alignment):")
    combos = [
        ("text:apple",  t_apple, "img:apple",  i_apple),
        ("text:stone",  t_stone, "img:stone",  i_stone),
        ("text:water",  t_water, "img:water",  i_water),
        ("text:apple",  t_apple, "img:stone",  i_stone),  # mismatch
    ]
    for n1, v1, n2, v2 in combos:
        sim = space.similarity(v1, v2)
        match = "✓ match" if "apple" in n1 and "apple" in n2 \
                          or "stone" in n1 and "stone" in n2 \
                          or "water" in n1 and "water" in n2 else "✗ mismatch"
        print(f"    {n1:15s} ↔ {n2:15s}: {sim:+.4f}  {match}")

    # After modality alignment (Procrustes)
    text_matrix  = np.stack([t_apple, t_stone, t_water])
    image_matrix = np.stack([i_apple, i_stone, i_water])
    R = space.align_modalities(image_matrix, text_matrix)

    print("\n  Cross-modal similarities (after Procrustes alignment):")
    for n1, v1, n2, v2 in combos:
        if "img" in n2:
            v2_aligned = space.l2(v2 @ R)
        else:
            v2_aligned = v2
        sim = space.similarity(v1, v2_aligned)
        print(f"    {n1:15s} ↔ {n2:15s}: {sim:+.4f}")


# ═══════════════════════════════════════════════════════
# 5. Reasoning Loop
# ═══════════════════════════════════════════════════════
def demo_reasoning():
    sep("5. Reasoning Loop — Self-Correction")

    from cognifield.encoder.text_encoder      import TextEncoder
    from cognifield.latent_space.frequency_space import FrequencySpace
    from cognifield.memory.memory_store       import MemoryStore
    from cognifield.reasoning.reasoning_engine import ReasoningEngine

    space  = FrequencySpace(dim=64)
    memory = MemoryStore(dim=64)
    enc    = TextEncoder(dim=64); enc.fit()
    engine = ReasoningEngine(space, memory, max_retries=6,
                              threshold=0.65)

    # Pre-load memory with known concepts
    known = ["apple", "fruit", "food", "eat", "red apple",
             "pick up stone", "move object", "observe environment",
             "the agent learns", "curiosity drives exploration"]
    for text in known:
        vec = enc.encode(text)
        memory.store(vec, label=text, modality="text")

    print(f"\n  Memory loaded: {len(memory)} patterns")
    print(f"\n  Reasoning sessions:")

    sessions = [
        ("I eat apple",          "apple fruit food"),
        ("pick up the stone",    "pick up stone"),
        ("the curious agent",    "curiosity drives exploration"),
        ("xzqwpvr unknown blob", "apple fruit food"),  # hard case
    ]

    for input_text, target_text in sessions:
        inp_vec = enc.encode(input_text)
        tgt_vec = enc.encode(target_text)
        result  = engine.reason(inp_vec, tgt_vec, verbose=False)
        status  = "✓ SUCCESS" if result.success else f"✗ {result.error_type.value}"
        print(f"    input='{input_text[:30]}'")
        print(f"    target='{target_text[:30]}'")
        print(f"    → score={result.score:.3f}  retries={result.n_retries}"
              f"  strategy='{result.strategy_used}'  {status}")
        print()

    print(f"  Overall success rate: {engine.success_rate():.1%}")
    print(f"\n  Strategy win rates:")
    for row in engine.strategy_report()[:4]:
        print(f"    {row['strategy']:22s} wins={row['wins']}/{row['tries']}"
              f"  rate={row['rate']:.1%}")


# ═══════════════════════════════════════════════════════
# 6. Curiosity Engine
# ═══════════════════════════════════════════════════════
def demo_curiosity():
    sep("6. Curiosity Engine — Novelty Detection")

    from cognifield.encoder.text_encoder   import TextEncoder
    from cognifield.memory.memory_store    import MemoryStore
    from cognifield.curiosity.curiosity_engine import CuriosityEngine
    from cognifield.latent_space.frequency_space import FrequencySpace

    space  = FrequencySpace(dim=64)
    memory = MemoryStore(dim=64)
    enc    = TextEncoder(dim=64); enc.fit()
    engine = CuriosityEngine(space, memory, novelty_threshold=0.35)

    # Load familiar concepts
    familiar = ["apple", "stone", "book", "water", "hammer",
                "the agent picks up the apple",
                "move to position five",
                "observe the environment carefully"]
    for text in familiar:
        vec = enc.encode(text)
        memory.store(vec, label=text, modality="text",
                     allow_duplicate=True)    # force store all

    print(f"\n  Known memory: {len(memory)} patterns\n")

    # Test a mix of familiar and novel inputs
    test_inputs = [
        ("apple",                          "familiar"),
        ("the agent picks up the apple",   "familiar"),
        ("a blue flamingo dances at midnight", "novel"),
        ("quantum entanglement in brains", "novel"),
        ("stone hammer tool",              "partially known"),
        ("xyz pqr wuv",                    "completely novel"),
    ]

    print(f"  {'Input':42s} | expected    | novelty | triggered")
    print(f"  {'─'*42} | {'─'*11} | {'─'*7} | {'─'*9}")
    for text, expected in test_inputs:
        vec     = enc.encode(text)
        novelty = engine.detect_novelty(vec)
        trig    = "YES ⚡" if novelty >= 0.35 else "no"
        bar     = "▓" * int(novelty * 10)
        print(f"  {text[:42]:42s} | {expected:11s} | {novelty:.3f} {bar:<10s}| {trig}")
        if novelty >= 0.35:
            report = engine.trigger_exploration(vec, raw_input=text, modality="text")
            print(f"    → hypothesis: '{report.hypothesis}'")
            print(f"    → nearest known: '{report.nearest_known}' "
                  f"(sim={report.nearest_sim:.3f})")

    print(f"\n  Total explorations: {engine.n_explorations}")
    print(f"  Summary: {engine.exploration_summary()}")


# ═══════════════════════════════════════════════════════
# 7. Language Checker
# ═══════════════════════════════════════════════════════
def demo_language_checker():
    sep("7. Language Checker — Grammar & Semantic Scoring")

    from cognifield.language.structure_checker import StructureChecker

    checker = StructureChecker()

    test_texts = [
        "I eat apple in the morning.",
        "The agent picks up the red apple from the ground.",
        "the the apple is is red red",        # grammar issues
        "runs quickly without any reason",    # missing subject
        "a an the a the",                     # no content words
        "water flows down the mountain.",
        "xqz pqr wvl",                        # nonsense
        "The curious agent explores novel inputs and learns from feedback.",
    ]

    print(f"\n  {'Text':50s} | grammar | semantic | overall | valid")
    print(f"  {'─'*50} | {'─'*7} | {'─'*8} | {'─'*7} | {'─'*5}")
    for text in test_texts:
        report = checker.check(text)
        v = "✓" if report.is_valid else "✗"
        print(f"  {text[:50]:50s} | {report.grammar_score:.3f}   | "
              f"{report.semantic_score:.3f}    | {report.overall_score:.3f}   | {v}")
        if report.issues:
            for issue in report.issues[:1]:
                print(f"      ↳ {issue}")


# ═══════════════════════════════════════════════════════
# 8. Environment Interaction
# ═══════════════════════════════════════════════════════
def demo_environment():
    sep("8. Environment Interaction")

    from cognifield.environment.simple_env import SimpleEnv

    env = SimpleEnv(seed=42)

    print(f"\n  World objects: {', '.join(env.object_names)}")
    print(f"  Agent start: {env._agent_pos}\n")

    actions = [
        ("observe",  ()),
        ("move",     (4, 3)),
        ("observe",  ()),
        ("pick",     ("apple",)),
        ("pick",     ("stone",)),
        ("inspect",  ("apple",)),
        ("eat",      ("apple",)),
        ("drop",     ("stone",)),
        ("observe",  ()),
    ]

    total_reward = 0.0
    for action, args in actions:
        fb = env.step(action, *args)
        total_reward += fb.get("reward", 0)
        status = "✓" if fb["success"] else "✗"
        print(f"  {status} {action:8s} {str(args):18s} → {fb['message'][:60]}")
        print(f"           reward={fb['reward']:+.2f}  total={total_reward:+.2f}")

    print(f"\n  Final state: {env}")
    print(f"  Stats: {env.stats()}")


# ═══════════════════════════════════════════════════════
# 9. Full Agent Loop
# ═══════════════════════════════════════════════════════
def demo_full_agent():
    sep("9. Full Agent Loop — I eat apple (multimodal)")

    from cognifield.agent.agent import CogniFieldAgent, AgentConfig
    from cognifield.environment.simple_env import SimpleEnv

    env   = SimpleEnv(seed=99)
    agent = CogniFieldAgent(
        config=AgentConfig(dim=64, novelty_threshold=0.35, seed=42),
        env=env,
    )

    # ── Phase A: Load known concepts via text ──
    print("  [Phase A] Loading known concepts...")
    known_texts = [
        ("apple red fruit food",       "apple"),
        ("stone heavy rock grey",      "stone"),
        ("water liquid clear drink",   "water"),
        ("pick up grab hold object",   "pick"),
        ("eat consume food hunger",    "eat"),
        ("observe look see around",    "observe"),
        ("the agent explores world",   "agent"),
    ]
    for text, label in known_texts:
        agent.step(text, modality="text", label=label,
                   target_text=label, verbose=False)

    print(f"  Memory loaded: {len(agent.memory)} patterns\n")

    # ── Phase B: Key example — text + image linking ──
    print("  [Phase B] Cross-modal: 'I eat apple' + apple image")

    # Text input
    rec_text = agent.observe("I eat apple", modality="text", label="eat_apple")
    print(f"    Text vec  spectral_mean = {rec_text.vector[:4].round(3)}")

    # Synthetic apple image (red-dominant)
    apple_img = np.zeros((64, 64, 3), dtype=np.float32)
    apple_img[:, :, 0] = 0.85
    apple_img[:, :, 1] = 0.12
    apple_img[:, :, 2] = 0.05
    rec_img = agent.observe(apple_img, modality="image", label="apple_image")
    print(f"    Image vec spectral_mean = {rec_img.vector[:4].round(3)}")

    sim_before = agent.space.similarity(rec_text.vector, rec_img.vector)
    sim_after  = agent.link_modalities("I eat apple", rec_img.vector, "image")
    print(f"    Similarity before alignment: {sim_before:+.4f}")
    print(f"    Similarity after  alignment: {sim_after:+.4f}")

    # ── Phase C: Environment loop ──
    print("\n  [Phase C] Agent in environment")

    env_steps = [
        ("I observe the environment around me", "observe", "observe", ()),
        ("move to where the apple is",         "move",    "move",    (3, 4)),
        ("pick up the apple from the ground",  "pick",    "pick",    ("apple",)),
        ("I want to eat this apple now",       "eat",     "eat",     ("apple",)),
        ("what do I see around me now",        "observe", "observe", ()),
    ]

    print(f"\n  {'Step':4s} | {'Input text':38s} | score | reward | novel")
    print(f"  {'─'*4} | {'─'*38} | {'─'*5} | {'─'*6} | {'─'*5}")
    for text, label, action, args in env_steps:
        s = agent.step(
            raw_input=text, modality="text", label=label,
            target_text=label,
            action=action, action_args=args,
            verbose=False,
        )
        novel = "⚡" if s.novel else "–"
        env_r = f"{s.env_reward:+.2f}" if s.env_reward is not None else "  N/A"
        print(f"  {s.step:4d} | {text[:38]:38s} | {s.reasoning_score:.3f} | {env_r:6s} | {novel}")

    # ── Phase D: Curiosity — inject novel inputs ──
    print("\n  [Phase D] Curiosity — novel inputs")
    novel_inputs = [
        "a purple dragon sings opera in quantum foam",
        "stone",              # familiar
        "holographic rainbows in the silicon mind",
    ]
    for text in novel_inputs:
        s = agent.step(text, modality="text", label="novel_test")
        marker = "⚡ NOVEL" if s.novel else "  known"
        print(f"    {marker}  '{text[:55]}'  novelty triggers={agent.curiosity.n_explorations}")

    # ── Phase E: Memory recall ──
    print("\n  [Phase E] Memory recall")
    queries = ["apple", "eat fruit", "pick up object"]
    for q in queries:
        results = agent.recall(q, k=3)
        top = [f"'{lbl}'({sim:.3f})" for sim, lbl in results[:3]]
        print(f"    query='{q}' → {', '.join(top)}")

    # ── Summary ──
    print("\n  ── Agent Summary ──")
    summ = agent.summary()
    print(f"    Steps:           {summ['steps']}")
    print(f"    Memory size:     {summ['memory_size']}")
    print(f"    Reasoning rate:  {summ['reasoning_success']}")
    print(f"    Curiosity expl.: {summ['curiosity_explorations']}")
    print(f"    Loss summary:    {summ['loss']}")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║   CogniField — Frequency-Space Cognitive Demo       ║")
    print("╚══════════════════════════════════════════════════════╝")

    demo_frequency_space()
    demo_text_encoding()
    demo_image_encoding()
    demo_multimodal_linking()
    demo_reasoning()
    demo_curiosity()
    demo_language_checker()
    demo_environment()
    demo_full_agent()

    print("\n" + "═"*56)
    print("  Demo complete.")
    print("═"*56)
