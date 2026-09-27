# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

from backend import config
from backend.namenode import NameNode
from backend.util import needs_recovery


def node(node_id, rack, free=1024, state="LIVE", capacity=2048, used=0):
    return {
        "node_id": node_id,
        "rack": rack,
        "state": state,
        "url": f"http://127.0.0.1/{node_id}",
        "storage": {"capacity": capacity, "used": used, "free": free},
    }


def block(bid, replicas, desired=3, genstamp=1):
    return {
        "id": bid,
        "size": 10,
        "checksum": "sha",
        "genstamp": genstamp,
        "desired": desired,
        "replicas": {
            nid: {"genstamp": genstamp, "checksum": "sha", "size": 10,
                  "state": "ok"}
            for nid in replicas
        },
    }


class RackPlacementTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._old_session = config.SESSION_DIR
        config.SESSION_DIR = str(root / "sessions")
        self.nn = NameNode(
            host="127.0.0.1", port=0,
            data_dir=str(root), meta_dir=str(root / "meta"))
        self.nn._init_blocks_doc()

    def tearDown(self):
        config.SESSION_DIR = self._old_session
        self.tmp.cleanup()

    def _install_nodes(self, *nodes):
        self.nn.nodes = {n["node_id"]: n for n in nodes}

    def _install_block(self, *replicas, bid="blk-test", desired=3):
        blk = block(bid, replicas, desired)
        self.nn.meta.get("blocks")["blocks"][bid] = blk
        return blk

    def _drain(self, node_id):
        with self.nn.cmd_lock:
            cmds = self.nn.pending_commands.pop(node_id, [])
        return cmds

    def test_two_of_three_replicas_triggers_recovery(self):
        self.assertEqual(config.RECOVERY_TRIGGER, "desired")
        self.assertTrue(needs_recovery(
            2, 3, config.MIN_REPLICATION, config.RECOVERY_TRIGGER))

    def test_recovery_target_avoids_racks_with_existing_good_replicas(self):
        self._install_nodes(node("dn1", "rack-1"), node("dn2", "rack-2"),
                            node("dn3", "rack-3"), node("dn4", "rack-1"))
        blk = self._install_block("dn1", "dn4", bid="blk-a")
        self.nn.under_replicated["blk-a"] = {"since": 0, "attempts": 0}

        self.nn._schedule_recovery_once()
        commands = []
        for nid in ("dn2", "dn3"):
            commands.extend(self._drain(nid))
        self.assertEqual(self._drain("dn4"), [])
        cmd = commands[0]

        self.assertEqual(cmd["type"], "replicate")
        self.assertEqual(cmd["block_id"], "blk-a")
        self.assertEqual(blk["desired"], 3)

    def test_recovery_can_share_rack_when_too_few_distinct_racks(self):
        self._install_nodes(node("dn1", "rack-1"), node("dn4", "rack-1"),
                            node("dn2", "rack-2"))
        self._install_block("dn1", bid="blk-b")
        self.nn.under_replicated["blk-b"] = {"since": 0, "attempts": 0}

        self.nn._schedule_recovery_once()
        destinations = []
        for nid in ("dn2", "dn4"):
            destinations.extend(c for c in self._drain(nid)
                                if c["type"] == "replicate")

        self.assertEqual({c["block_id"] for c in destinations}, {"blk-b"})
        self.assertEqual(len(destinations), 2)

    def test_rebalance_moves_same_rack_replica_after_recovery(self):
        self._install_nodes(node("dn1", "rack-1"), node("dn4", "rack-1"),
                            node("dn2", "rack-2"), node("dn3", "rack-3"))
        self._install_block("dn1", "dn4", "dn2", bid="blk-c")

        self.nn._schedule_rack_rebalancing_once()
        cmd = self._drain("dn3")[0]
        self.assertEqual(cmd["type"], "replicate")
        self.assertEqual(cmd["block_id"], "blk-c")
        self.assertEqual(cmd["migration_id"],
                         self.nn.rack_adjustments["blk-c"]["id"])
        self.assertIn(self.nn.rack_adjustments["blk-c"]["src"],
                      {"dn1", "dn4"})

        self.nn._handle_node_event(
            "dn3", {"type": "replicate_done", "block_id": "blk-c",
                    "migration_id": cmd["migration_id"], "genstamp": 1,
                    "checksum": "sha", "size": 10})
        src = self.nn.rack_adjustments["blk-c"]["src"]
        self.assertIn("dn3",
                      self.nn.meta.get("blocks")["blocks"]["blk-c"]["replicas"])
        self.assertEqual(self._drain(src)[0]["type"], "delete")

        self.nn._handle_node_event(
            src, {"type": "deleted", "block_id": "blk-c"})
        final = self.nn.meta.get("blocks")["blocks"]["blk-c"]
        other = ({"dn1", "dn4"} - {src}).pop()
        self.assertEqual(set(final["replicas"]), {"dn2", "dn3", other})

    def test_no_violation_when_topology_cannot_provide_more_racks(self):
        self._install_nodes(node("dn1", "rack-1"), node("dn4", "rack-1"),
                            node("dn2", "rack-2"))
        blk = self._install_block("dn1", "dn4", "dn2", bid="blk-d")

        self.assertIsNone(self.nn._rack_violation(
            blk, list(self.nn.nodes.values()),
            *self.nn._block_rack_layout(blk, list(self.nn.nodes.values()))))
        self.nn._schedule_rack_rebalancing_once()
        self.assertNotIn("blk-d", self.nn.rack_adjustments)

    def test_surplus_replica_is_deleted_from_crowded_rack(self):
        self._install_nodes(node("dn1", "rack-1"), node("dn4", "rack-1"),
                            node("dn2", "rack-2"), node("dn3", "rack-3"))
        self._install_block("dn1", "dn4", "dn2", "dn3", bid="blk-e",
                            desired=3)

        self.nn._schedule_rack_rebalancing_once()
        # 超量副本必然来自 rack-1；删除后达到 3 机架、3 副本。
        deleted = [n for n in ("dn1", "dn4") if self._drain(n)]
        self.assertEqual(len(deleted), 1)
        remaining = set(
            self.nn.meta.get("blocks")["blocks"]["blk-e"]["replicas"])
        self.assertEqual(len(remaining), 3)
        self.assertIn(deleted[0], {"dn1", "dn4"})
        self.assertEqual(len(remaining & {"dn1", "dn4"}), 1)


if __name__ == "__main__":
    unittest.main()
