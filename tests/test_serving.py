from hr_policy_navigator.serving import RateLimiter, ResponseCache


def test_rate_limiter_window():
    t = [0.0]
    rl = RateLimiter(2, 60, clock=lambda: t[0])
    assert rl.check("a")[0] and rl.check("a")[0]
    ok, wait = rl.check("a")
    assert not ok and wait > 0
    assert rl.check("b")[0]          # independent keys
    t[0] = 61
    assert rl.check("a")[0]          # window expired


def test_cache_ttl_lru_and_only_success():
    t = [0.0]
    c = ResponseCache(ttl_s=10, max_items=2, clock=lambda: t[0])
    c.put("How do I enroll?", "s", {"status": "success", "answer": "x"})
    assert c.get("how do i  enroll", "s")["cached"] is True      # normalised key
    assert c.get("how do i enroll", "other") is None             # strategy is part of key
    c.put("q", "s", {"status": "refused"}); assert c.get("q", "s") is None
    t[0] = 11; assert c.get("how do i enroll", "s") is None      # expired
    for i in range(3):
        c.put(f"q{i}", "s", {"status": "success", "answer": str(i)})
    assert c.get("q0", "s") is None and c.get("q2", "s")         # LRU eviction
