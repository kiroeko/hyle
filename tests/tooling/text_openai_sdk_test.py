"""Exercise the Responses adapter through the optional OpenAI Python SDK."""
import importlib.util
import threading
import unittest

from engine.text.app import TextApiServer
from engine.text.runtime import TextEngine
from engine.text.worker_client import TokenEvent
from tests.tooling.text_frontend_test import FakeCodec, FakeWorker


SDK_AVAILABLE = importlib.util.find_spec('openai') is not None and importlib.util.find_spec('httpx') is not None


@unittest.skipUnless(SDK_AVAILABLE, 'OpenAI Python SDK is not installed')
class ResponsesSdkTest(unittest.TestCase):
    def setUp(self):
        import httpx
        from openai import OpenAI
        self.worker = FakeWorker()
        self.server = TextApiServer(('127.0.0.1', 0), TextEngine(self.worker, FakeCodec(), 'hyle'), True)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.client = OpenAI(base_url=f'http://127.0.0.1:{self.server.server_port}/v1', api_key='local-test',
                             max_retries=0, http_client=httpx.Client(trust_env=False),
                             _strict_response_validation=True)
        self.params = dict(model='hyle', input='x', max_output_tokens=3, store=False)

    def tearDown(self):
        self.client.close()
        self.server.shutdown()
        self.thread.join(timeout=3)
        self.server.server_close()

    def test_strict_sync_and_stream_helper_agree(self):
        response = self.client.responses.create(**self.params)
        self.assertEqual((response.status, response.output_text, response.usage.output_tokens), ('completed', 'a', 2))
        with self.client.responses.stream(**self.params) as stream:
            events = list(stream)
            final = stream.get_final_response()
        self.assertEqual([event.sequence_number for event in events], list(range(len(events))))
        self.assertEqual(events[-1].type, 'response.completed')
        self.assertEqual((final.status, final.output_text), ('completed', 'a'))
        self.assertEqual(''.join(event.delta for event in events if event.type == 'response.output_text.delta'), 'a')

    def test_sdk_receives_rejection_and_typed_stream_error(self):
        from openai import BadRequestError
        with self.assertRaises(BadRequestError) as error:
            self.client.responses.create(**self.params, temperature=.8)
        self.assertEqual(error.exception.status_code, 400)

        class ErrorWorker(FakeWorker):
            def submit(self, tokens, max_output_tokens, *, stop_ids=()):
                request = super().submit(tokens, max_output_tokens, stop_ids=stop_ids)
                request.events = iter([TokenEvent('error', message='synthetic failure', error_kind='model')])
                return request

        self.server.engine.worker = ErrorWorker()
        events = list(self.client.responses.create(**self.params, stream=True))
        self.assertEqual(events[-1].type, 'error')
        self.assertEqual((events[-1].code, events[-1].message), ('server_error', 'synthetic failure'))
        self.assertEqual([event.sequence_number for event in events], list(range(len(events))))

    def test_sdk_raw_stream_exposes_incomplete_terminal(self):
        class LimitedWorker(FakeWorker):
            def submit(self, tokens, max_output_tokens, *, stop_ids=()):
                request = super().submit(tokens, max_output_tokens, stop_ids=stop_ids)
                request.events = iter([TokenEvent('token', ord('a')), TokenEvent('token', ord('b')),
                                       TokenEvent('token', ord('c')), TokenEvent('completed')])
                return request

        self.server.engine.worker = LimitedWorker()
        events = list(self.client.responses.create(**self.params, stream=True))
        self.assertEqual(events[-1].type, 'response.incomplete')
        self.assertEqual(events[-1].response.status, 'incomplete')
        self.assertEqual(events[-1].response.incomplete_details.reason, 'max_output_tokens')
        self.assertEqual(events[-1].response.output_text, 'abc')


if __name__ == '__main__':
    unittest.main()
