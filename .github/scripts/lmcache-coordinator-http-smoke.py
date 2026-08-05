#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Exercise the LMCache coordinator key directory through its HTTP API.

The cases mirror every behavior from LMCache PR #4275 that is reachable through
the public coordinator API. ``KeyDirectory.drop_instance`` is intentionally not
covered: coordinator instance deregistration was not wired to that internal
method by the PR, so no public HTTP operation exercises it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import argparse
import json
import urllib.error
import urllib.request


JSON = dict[str, Any]


class CoordinatorClient:
    """Small standard-library JSON client for the coordinator API."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        body: JSON | None = None,
        expected_status: int = 200,
    ) -> Any:
        encoded = None
        headers: dict[str, str] = {}
        if body is not None:
            encoded = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=encoded,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw = exc.read()
        except urllib.error.URLError as exc:
            raise AssertionError(f"{method} {path} failed: {exc}") from exc

        text = raw.decode("utf-8", errors="replace")
        if status != expected_status:
            raise AssertionError(
                f"{method} {path}: expected HTTP {expected_status}, got "
                f"{status}: {text}"
            )
        if not raw:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{method} {path}: response is not JSON: {text}"
            ) from exc

    def health(self) -> Any:
        return self.request("GET", "/healthz")

    def post_events(
        self, batches: list[JSON], expected_status: int = 200
    ) -> Any:
        return self.request(
            "POST",
            "/directory/events",
            {"batches": batches},
            expected_status,
        )

    def lookup(self, keys: list[JSON]) -> JSON:
        return self.request("POST", "/directory/lookup", {"keys": keys})

    def lookup_tokens(
        self, body: JSON, expected_status: int = 200
    ) -> Any:
        return self.request(
            "POST", "/directory/lookup_tokens", body, expected_status
        )

    def stats(self) -> JSON:
        return self.request("GET", "/directory/stats")


def key(
    chunk_hash_hex: str,
    model_name: str,
    kv_rank: int = 0,
    object_group_id: int = 0,
    cache_salt: str = "",
) -> JSON:
    return {
        "chunk_hash_hex": chunk_hash_hex,
        "model_name": model_name,
        "kv_rank": kv_rank,
        "object_group_id": object_group_id,
        "cache_salt": cache_salt,
    }


def entry(
    object_key: JSON,
    size_bytes: int = 0,
    content_hash_hex: str = "",
) -> JSON:
    return {
        "key": object_key,
        "size_bytes": size_bytes,
        "content_hash_hex": content_hash_hex,
    }


def batch(
    instance_id: str,
    incarnation: int,
    seq: int,
    entries: list[JSON],
    event_type: str = "store",
    tier: str = "l1",
    backend: str = "dram",
    timestamp: float = 0.0,
) -> JSON:
    return {
        "instance_id": instance_id,
        "incarnation": incarnation,
        "seq": seq,
        "event_type": event_type,
        "tier": tier,
        "backend": backend,
        "entries": entries,
        "ts": timestamp,
    }


def assert_counts(
    response: JSON,
    *,
    applied: int = 0,
    duplicates: int = 0,
    stale: int = 0,
) -> None:
    expected = {
        "applied": applied,
        "duplicates": duplicates,
        "stale": stale,
    }
    assert response == expected, f"event counts: expected {expected}, got {response}"


def assert_placement(
    placement: JSON,
    *,
    instance_id: str,
    incarnation: int,
    tier: str,
    backend: str,
    size_bytes: int,
) -> None:
    expected = {
        "instance_id": instance_id,
        "incarnation": incarnation,
        "tier": tier,
        "backend": backend,
        "size_bytes": size_bytes,
    }
    actual = {field: placement.get(field) for field in expected}
    assert actual == expected, f"placement: expected {expected}, got {placement}"


def assert_no_mutation(client: CoordinatorClient, action: Callable[[], Any]) -> None:
    before = client.stats()
    action()
    after = client.stats()
    assert after == before, f"rejected request mutated directory: {before} -> {after}"


def run_case(name: str, test: Callable[[], None]) -> None:
    print(f"--- {name}")
    test()
    print(f"PASS: {name}")


def run_suite(client: CoordinatorClient) -> int:
    completed = 0

    def case(name: str, test: Callable[[], None]) -> None:
        nonlocal completed
        run_case(name, test)
        completed += 1

    def store_lookup_unknown_and_order() -> None:
        instance = "store-node"
        known = key("01010101", "smoke-store")
        unknown = key("ffffffff", "smoke-store")
        response = client.post_events(
            [batch(instance, 1, 1, [entry(known, 1024)])]
        )
        assert_counts(response, applied=1)

        results = client.lookup([unknown, known])["results"]
        assert [result["key"] for result in results] == [unknown, known]
        assert results[0]["placements"] == []
        [placement] = results[1]["placements"]
        assert_placement(
            placement,
            instance_id=instance,
            incarnation=1,
            tier="l1",
            backend="dram",
            size_bytes=1024,
        )

        state = client.stats()["instances"][instance]
        assert state["incarnation"] == 1
        assert state["last_seq"] == 1
        assert state["gap_detected"] is False
        assert state["num_keys"] == 1

    case("store, lookup, unknown key, and request order", store_lookup_unknown_and_order)

    def restore_updates_size() -> None:
        instance = "restore-node"
        object_key = key("02020202", "smoke-restore")
        assert_counts(
            client.post_events(
                [batch(instance, 1, 1, [entry(object_key, 100)])]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [batch(instance, 1, 2, [entry(object_key, 200)])]
            ),
            applied=1,
        )
        [result] = client.lookup([object_key])["results"]
        [placement] = result["placements"]
        assert placement["size_bytes"] == 200
        assert client.stats()["instances"][instance]["num_keys"] == 1

    case("re-store updates size without a duplicate", restore_updates_size)

    def same_key_multiple_instances() -> None:
        object_key = key("03030303", "smoke-multi-instance")
        assert_counts(
            client.post_events(
                [batch("multi-node-b", 1, 1, [entry(object_key, 200)])]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [batch("multi-node-a", 1, 1, [entry(object_key, 100)])]
            ),
            applied=1,
        )
        [result] = client.lookup([object_key])["results"]
        placements = result["placements"]
        assert [placement["instance_id"] for placement in placements] == [
            "multi-node-a",
            "multi-node-b",
        ]

    case("same key on two instances", same_key_multiple_instances)

    def same_key_multiple_tiers() -> None:
        instance = "multi-tier-node"
        object_key = key("04040404", "smoke-multi-tier")
        assert_counts(
            client.post_events(
                [batch(instance, 1, 1, [entry(object_key, 100)])]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [
                    batch(
                        instance,
                        1,
                        2,
                        [entry(object_key, 200)],
                        tier="l2",
                        backend="fs",
                    )
                ]
            ),
            applied=1,
        )
        [result] = client.lookup([object_key])["results"]
        assert [
            (placement["tier"], placement["backend"])
            for placement in result["placements"]
        ] == [("l1", "dram"), ("l2", "fs")]

    case("same key on two tiers", same_key_multiple_tiers)

    def deletion_semantics() -> None:
        delete_instance = "delete-node"
        deleted_key = key("05050505", "smoke-delete")
        assert_counts(
            client.post_events(
                [batch(delete_instance, 1, 1, [entry(deleted_key, 100)])]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [
                    batch(
                        delete_instance,
                        1,
                        2,
                        [entry(deleted_key)],
                        event_type="delete",
                    )
                ]
            ),
            applied=1,
        )
        assert client.lookup([deleted_key])["results"][0]["placements"] == []
        stats = client.stats()
        assert stats["instances"][delete_instance]["num_keys"] == 0

        keep_instance = "delete-one-tier-node"
        tiered_key = key("06060606", "smoke-delete-one-tier")
        assert_counts(
            client.post_events(
                [batch(keep_instance, 1, 1, [entry(tiered_key, 100)])]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [
                    batch(
                        keep_instance,
                        1,
                        2,
                        [entry(tiered_key, 200)],
                        tier="l2",
                        backend="fs",
                    )
                ]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [
                    batch(
                        keep_instance,
                        1,
                        3,
                        [entry(tiered_key)],
                        event_type="delete",
                    )
                ]
            ),
            applied=1,
        )
        placements = client.lookup([tiered_key])["results"][0]["placements"]
        assert [(p["tier"], p["backend"]) for p in placements] == [
            ("l2", "fs")
        ]
        assert stats_for(client, keep_instance)["num_keys"] == 1

        unknown_key = key("07070707", "smoke-delete-unknown")
        before = client.stats()
        assert_counts(
            client.post_events(
                [
                    batch(
                        "delete-unknown-node",
                        1,
                        1,
                        [entry(unknown_key)],
                        event_type="delete",
                    )
                ]
            ),
            applied=1,
        )
        after = client.stats()
        assert after["num_keys"] == before["num_keys"]
        assert after["num_placements"] == before["num_placements"]
        assert client.lookup([unknown_key])["results"][0]["placements"] == []

    case("delete, tier-preserving delete, and unknown delete", deletion_semantics)

    def access_does_not_create() -> None:
        instance = "access-node"
        object_key = key("08080808", "smoke-access")
        before = client.stats()
        assert_counts(
            client.post_events(
                [
                    batch(
                        instance,
                        1,
                        1,
                        [entry(object_key)],
                        event_type="access",
                        backend="",
                    )
                ]
            ),
            applied=1,
        )
        after = client.stats()
        assert after["num_keys"] == before["num_keys"]
        assert after["num_placements"] == before["num_placements"]
        assert after["instances"][instance]["num_keys"] == 0
        assert client.lookup([object_key])["results"][0]["placements"] == []

    case("access does not create a record", access_does_not_create)

    def duplicate_and_older_sequences() -> None:
        duplicate_instance = "duplicate-node"
        duplicate_key = key("09090909", "smoke-duplicate")
        assert_counts(
            client.post_events(
                [
                    batch(
                        duplicate_instance,
                        1,
                        1,
                        [entry(duplicate_key, 100)],
                    )
                ]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [
                    batch(
                        duplicate_instance,
                        1,
                        1,
                        [entry(duplicate_key, 999)],
                    )
                ]
            ),
            duplicates=1,
        )
        placement = client.lookup([duplicate_key])["results"][0]["placements"][0]
        assert placement["size_bytes"] == 100

        older_instance = "older-seq-node"
        older_key = key("0a0a0a0a", "smoke-older-seq")
        assert_counts(
            client.post_events(
                [batch(older_instance, 1, 1, [entry(older_key, 100)])]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [
                    batch(
                        older_instance,
                        1,
                        2,
                        [entry(older_key)],
                        event_type="delete",
                    )
                ]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [batch(older_instance, 1, 1, [entry(older_key, 999)])]
            ),
            duplicates=1,
        )
        assert client.lookup([older_key])["results"][0]["placements"] == []

    case("duplicate and replayed older sequences", duplicate_and_older_sequences)

    def mixed_batch_counts() -> None:
        instance = "mixed-count-node"
        original = key("0b0b0b0b", "smoke-mixed-count")
        stale_key = key("0c0c0c0c", "smoke-mixed-count")
        fresh_key = key("0d0d0d0d", "smoke-mixed-count")
        assert_counts(
            client.post_events(
                [batch(instance, 2, 1, [entry(original, 100)])]
            ),
            applied=1,
        )
        response = client.post_events(
            [
                batch(instance, 2, 1, [entry(original, 999)]),
                batch(instance, 1, 9, [entry(stale_key, 200)]),
                batch(instance, 2, 2, [entry(fresh_key, 300)]),
            ]
        )
        assert_counts(response, applied=1, duplicates=1, stale=1)
        results = client.lookup([original, stale_key, fresh_key])["results"]
        assert results[0]["placements"][0]["size_bytes"] == 100
        assert results[1]["placements"] == []
        assert results[2]["placements"][0]["size_bytes"] == 300

    case("mixed applied, duplicate, and stale batch counts", mixed_batch_counts)

    def contiguous_and_gapped_sequences() -> None:
        contiguous_instance = "contiguous-node"
        key_one = key("0e0e0e0e", "smoke-contiguous")
        key_two = key("0f0f0f0f", "smoke-contiguous")
        assert_counts(
            client.post_events(
                [batch(contiguous_instance, 1, 1, [entry(key_one, 100)])]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [batch(contiguous_instance, 1, 2, [entry(key_two, 200)])]
            ),
            applied=1,
        )
        contiguous = stats_for(client, contiguous_instance)
        assert contiguous["last_seq"] == 2
        assert contiguous["gap_detected"] is False

        gap_instance = "gap-node"
        gap_one = key("10101010", "smoke-gap")
        gap_two = key("11111111", "smoke-gap")
        assert_counts(
            client.post_events(
                [batch(gap_instance, 1, 1, [entry(gap_one, 100)])]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [batch(gap_instance, 1, 5, [entry(gap_two, 200)])]
            ),
            applied=1,
        )
        gap = stats_for(client, gap_instance)
        assert gap["last_seq"] == 5
        assert gap["gap_detected"] is True
        assert client.lookup([gap_two])["results"][0]["placements"]

    case("contiguous sequences and gap detection", contiguous_and_gapped_sequences)

    def incarnation_fencing() -> None:
        instance_a = "fence-node-a"
        instance_b = "fence-node-b"
        old_one = key("12121212", "smoke-fence")
        old_two = key("13131313", "smoke-fence")
        shared = key("14141414", "smoke-fence")
        new_key = key("15151515", "smoke-fence")
        assert_counts(
            client.post_events(
                [
                    batch(
                        instance_a,
                        1,
                        1,
                        [
                            entry(old_one, 100),
                            entry(old_two, 200),
                            entry(shared, 300),
                        ],
                    )
                ]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [batch(instance_b, 1, 1, [entry(shared, 400)])]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [batch(instance_a, 2, 1, [entry(new_key, 500)])]
            ),
            applied=1,
        )

        results = client.lookup([old_one, old_two, shared, new_key])["results"]
        assert results[0]["placements"] == []
        assert results[1]["placements"] == []
        assert [p["instance_id"] for p in results[2]["placements"]] == [
            instance_b
        ]
        [placement] = results[3]["placements"]
        assert placement["instance_id"] == instance_a
        assert placement["incarnation"] == 2
        state = stats_for(client, instance_a)
        assert state["incarnation"] == 2
        assert state["last_seq"] == 1
        assert state["gap_detected"] is False
        assert state["num_keys"] == 1

    case("new incarnation fencing preserves other instances", incarnation_fencing)

    def stale_incarnation() -> None:
        instance = "stale-node"
        current_key = key("16161616", "smoke-stale")
        stale_key = key("17171717", "smoke-stale")
        assert_counts(
            client.post_events(
                [batch(instance, 2, 1, [entry(current_key, 100)])]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [batch(instance, 1, 99, [entry(stale_key, 200)])]
            ),
            stale=1,
        )
        results = client.lookup([current_key, stale_key])["results"]
        assert results[0]["placements"]
        assert results[1]["placements"] == []
        state = stats_for(client, instance)
        assert state["incarnation"] == 2
        assert state["last_seq"] == 1

    case("stale incarnation rejection", stale_incarnation)

    def token_lookup() -> None:
        short = client.lookup_tokens(
            {
                "model_name": "smoke-token-short",
                "world_size": 1,
                "token_ids": list(range(10)),
                "cache_salt": "",
            }
        )
        assert short == {"chunks": 0, "results": []}

        body = {
            "model_name": "smoke-token-roundtrip",
            "world_size": 1,
            "token_ids": list(range(256)),
            "cache_salt": "",
        }
        first = client.lookup_tokens(body)
        assert first["chunks"] == 1
        assert len(first["results"]) == 1
        assert first["results"][0]["placements"] == []
        resolved_key = first["results"][0]["key"]
        assert_counts(
            client.post_events(
                [batch("token-node", 1, 1, [entry(resolved_key, 64)])]
            ),
            applied=1,
        )
        second = client.lookup_tokens(body)
        [placement] = second["results"][0]["placements"]
        assert placement["instance_id"] == "token-node"
        assert placement["size_bytes"] == 64

        invalid = {
            "model_name": "a@b",
            "world_size": 1,
            "token_ids": list(range(256)),
            "cache_salt": "",
        }
        client.lookup_tokens(invalid, expected_status=400)

    case("token lookup short, round-trip, and invalid model", token_lookup)

    def stats_count_keys_and_placements() -> None:
        before = client.stats()
        first = key("18181818", "smoke-stats")
        second = key("19191919", "smoke-stats")
        assert_counts(
            client.post_events(
                [
                    batch(
                        "stats-node-a",
                        1,
                        1,
                        [entry(first, 100), entry(second, 200)],
                    )
                ]
            ),
            applied=1,
        )
        assert_counts(
            client.post_events(
                [
                    batch(
                        "stats-node-b",
                        1,
                        1,
                        [entry(first, 300)],
                        tier="l2",
                        backend="fs",
                    )
                ]
            ),
            applied=1,
        )
        after = client.stats()
        assert after["num_keys"] == before["num_keys"] + 2
        assert after["num_placements"] == before["num_placements"] + 3
        assert after["instances"]["stats-node-a"]["num_keys"] == 2
        assert after["instances"]["stats-node-b"]["num_keys"] == 1

    case("stats count keys and placements", stats_count_keys_and_placements)

    def validation_errors() -> None:
        valid_key = key("1a1a1a1a", "smoke-validation")

        invalid_requests: list[tuple[str, JSON]] = [
            (
                "tier all",
                batch("invalid-tier", 1, 1, [entry(valid_key, 1)], tier="all"),
            ),
            (
                "sequence zero",
                batch("invalid-seq", 1, 0, [entry(valid_key, 1)]),
            ),
            (
                "negative size",
                batch("invalid-size", 1, 1, [entry(valid_key, -1)]),
            ),
            (
                "malformed key hash",
                batch(
                    "invalid-key-hash",
                    1,
                    1,
                    [entry(key("zz", "smoke-validation"), 1)],
                ),
            ),
            (
                "malformed content hash",
                batch(
                    "invalid-content-hash",
                    1,
                    1,
                    [entry(valid_key, 1, content_hash_hex="xyz")],
                ),
            ),
            (
                "store with empty backend",
                batch(
                    "invalid-store-backend",
                    1,
                    1,
                    [entry(valid_key, 1)],
                    backend="",
                ),
            ),
            (
                "delete with empty backend",
                batch(
                    "invalid-delete-backend",
                    1,
                    1,
                    [entry(valid_key)],
                    event_type="delete",
                    backend="",
                ),
            ),
        ]

        for label, invalid_batch in invalid_requests:
            assert_no_mutation(
                client,
                lambda invalid_batch=invalid_batch: client.post_events(
                    [invalid_batch], expected_status=422
                ),
            )
            print(f"  rejected as expected: {label}")

    case("event request validation and mutation safety", validation_errors)

    return completed


def stats_for(client: CoordinatorClient, instance_id: str) -> JSON:
    stats = client.stats()
    assert instance_id in stats["instances"], (
        f"instance {instance_id!r} missing from stats: {stats}"
    )
    return stats["instances"][instance_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coordinator-url",
        required=True,
        help="LMCache coordinator base URL, for example http://127.0.0.1:9300",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = CoordinatorClient(args.coordinator_url)
    client.health()
    completed = run_suite(client)
    final_stats = client.stats()
    print(
        "Coordinator HTTP smoke suite passed: "
        f"{completed} cases, {final_stats['num_keys']} keys, "
        f"{final_stats['num_placements']} placements"
    )
    print(
        "NOTE: KeyDirectory.drop_instance tests remain unit-only because no "
        "public coordinator HTTP endpoint invokes that internal method."
    )


if __name__ == "__main__":
    main()
