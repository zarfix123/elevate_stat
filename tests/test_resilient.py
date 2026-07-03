from elevate_stat import resilient


def test_for_each_processes_all_when_no_failures():
    seen = []
    failures = resilient.for_each([1, 2, 3], seen.append, label="n")
    assert seen == [1, 2, 3]
    assert failures == []


def test_for_each_continues_past_a_failing_item():
    seen = []

    def fn(x):
        if x == 2:
            raise ValueError("boom")
        seen.append(x)

    failures = resilient.for_each([1, 2, 3], fn, label="n")
    assert seen == [1, 3]  # 1 and 3 processed despite 2 failing
    assert len(failures) == 1
    assert failures[0][0] == 2
    assert isinstance(failures[0][1], ValueError)
