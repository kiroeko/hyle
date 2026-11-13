import queue
import threading
import time
import unittest
from types import SimpleNamespace

from engine.text.configuration import Selection, SessionPolicy
from engine.text.contracts import TokenEvent, WorkerError
from engine.text.sessions import SessionNotFound, TextSessions
from engine.text.worker_client import WorkerRequest


class Native:
    def __init__(self, worker, ids):
        self.worker, self.ids = worker, list(ids)
        self.events = queue.Queue()
        self.turn = 0
        self.retired = self.cancelled = False
        self.suffixes = []
        self.emit()

    def emit(self):
        self.events.put(TokenEvent('token', 3))
        self.events.put(TokenEvent('turn_end', turn=self.turn))

    def read(self, timeout=None): return self.events.get(timeout=timeout)
    def acknowledge(self): pass
    def append(self, ids, limit, **kwargs):
        self.suffixes.append(list(ids)); self.turn += 1; self.emit()
    def cancel(self): self.cancelled = True; return True
    def close(self): pass
    def retire(self, **kwargs):
        if self.worker.retirement_fails:
            raise WorkerError('timeout', 'no original terminal observation')
        self.cancel()
        self.retired = True
        return TokenEvent('cancelled')


class Worker:
    def __init__(self):
        self.requests = []
        self.domains = []
        self.retirement_fails = False
        self._domain = 0
    def allocate_cache_domain(self): self._domain += 1; return self._domain
    def open_session(self, ids, limit, **kwargs):
        request = Native(self, ids)
        self.requests.append(request); self.domains.append(kwargs['cache_sharing_domain'])
        return request


def complete(turn, key):
    assert turn.read().kind == 'token'
    turn.acknowledge()
    assert turn.read().kind == 'completed'
    turn.commit('3', {'id': key})
    turn.close()


class TextSessionsTest(unittest.TestCase):
    def owner(self, **kwargs):
        worker = Worker()
        owner = TextSessions(worker, SessionPolicy(**kwargs))
        self.addCleanup(owner.close)
        return owner, worker

    def test_exact_token_prefix_continues_original_pending_and_preserves_history(self):
        owner, worker = self.owner()
        first = owner.start('a', [{'role': 'user', 'content': '2'}], [1, 2], 1, Selection())
        complete(first, 'a')
        parent = owner.parent('a')
        next_turn = owner.start('b', [*parent.messages, {'role': 'user', 'content': '4'}],
                                [1, 2, 3, 4], 1, Selection(), parent=parent)
        self.assertEqual(next_turn.cached_tokens, 2)
        self.assertEqual(worker.requests[0].suffixes, [[4]])
        self.assertEqual(len(worker.requests), 1)
        complete(next_turn, 'b')
        self.assertEqual(owner.retrieve('a'), {'id': 'a'})
        self.assertEqual(len(parent.messages), 2)

    def test_forks_eviction_and_changed_profiles_rebuild_without_replacing_parent(self):
        owner, worker = self.owner(kv_capacity=1)
        first = owner.start('a', [], [1, 2], 1, Selection()); complete(first, 'a')
        parent = owner.parent('a')
        second = owner.start('b', [], [1, 2, 3, 4], 1, Selection(), parent=parent); complete(second, 'b')
        fork = owner.start('c', [], [1, 2, 3, 5], 1, Selection(1), parent=parent)
        self.assertTrue(worker.requests[0].retired)
        self.assertEqual(fork.cached_tokens, 0)
        self.assertEqual(len(worker.requests), 2)
        complete(fork, 'c')
        self.assertEqual(owner.retrieve('b'), {'id': 'b'})
        self.assertEqual(worker.domains[0], worker.domains[1])

    def test_idle_kv_reclaims_automatically_and_delete_waits_for_native_retirement(self):
        owner, worker = self.owner(ttl_seconds=.03)
        complete(owner.start('a', [], [1, 2], 1, Selection()), 'a')
        deadline = time.monotonic() + 2
        while not worker.requests[0].retired and time.monotonic() < deadline:
            threading.Event().wait(.01)
        self.assertTrue(worker.requests[0].retired)
        with self.assertRaises(SessionNotFound): owner.parent('a')
        complete(owner.start('b', [], [1, 2], 1, Selection()), 'b')
        owner.delete('b')
        self.assertTrue(worker.requests[1].retired)

    def test_active_budget_waits_and_close_wakes_waiting_ingress(self):
        owner, worker = self.owner(kv_capacity=1)
        first = owner.start('a', [], [1, 2], 1, Selection())
        arrived, errors = threading.Event(), []
        def pending():
            try: owner.start('b', [], [1, 2], 1, Selection())
            except Exception as error: errors.append(error)
            finally: arrived.set()
        thread = threading.Thread(target=pending); thread.start()
        self.assertFalse(arrived.wait(.03))
        owner.begin_close()
        self.assertTrue(arrived.wait(2)); thread.join()
        self.assertIsInstance(errors[0], WorkerError)
        first.close()
        self.assertTrue(worker.requests[0].retired)

    def test_retirement_timeout_never_recycles_native_capacity(self):
        worker = Worker(); owner = TextSessions(worker, SessionPolicy(kv_capacity=1))
        turn = owner.start('a', [], [1, 2], 1, Selection())
        worker.retirement_fails = True
        with self.assertRaises(WorkerError): turn.close()
        self.assertFalse(worker.requests[0].retired)
        self.assertFalse(owner.healthy)
        with self.assertRaises(WorkerError): owner.start('b', [], [1], 1, Selection())
        with self.assertRaisesRegex(WorkerError, 'unconfirmed'): owner.close()

    def test_trusted_shared_domain_is_opt_in_and_isolated_roots_never_share(self):
        for shared in (False, True):
            owner, worker = self.owner(shared_prefix=shared)
            complete(owner.start('a', [], [1], 1, Selection()), 'a')
            complete(owner.start('b', [], [1], 1, Selection()), 'b')
            self.assertEqual(worker.domains[0] == worker.domains[1], shared)

    def test_transport_retirement_drains_delivery_without_late_credit(self):
        events = queue.Queue(); sent, forgotten = [], []
        client = SimpleNamespace(_send=sent.append, _forget=forgotten.append,
                                 _read_event=lambda queue, timeout: queue.get(timeout=timeout))
        request = WorkerRequest(client, 9, events)
        events.put(TokenEvent('token', 1))
        self.assertEqual(request.read().token, 1)  # Deliberately leave credit held.
        events.put(TokenEvent('token', 2)); events.put(TokenEvent('cancelled'))
        self.assertEqual(request.retire(timeout=1).kind, 'cancelled')
        self.assertEqual(sent, ['C\t9'])
        self.assertEqual(forgotten, [9])

    def test_transport_timeout_is_not_a_retirement_observation(self):
        events = queue.Queue(); forgotten = []
        client = SimpleNamespace(_send=lambda frame: None, _forget=forgotten.append,
                                 _read_event=lambda queue, timeout: queue.get(timeout=timeout))
        request = WorkerRequest(client, 9, events)
        with self.assertRaisesRegex(WorkerError, 'unconfirmed'): request.retire(timeout=.01)
        self.assertEqual(forgotten, [])


if __name__ == '__main__': unittest.main()
