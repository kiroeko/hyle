"""Shut down a real worker with an active Responses stream."""
import argparse
import socket
import threading

from engine.text.app import ResponsesServer
from engine.text.runtime import TextEngine
from engine.text.worker_client import WorkerClient
from tests.tooling.text_worker_error_integration_test import FixtureCodec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--port', type=int, default=2048)
    parser.add_argument('--metrics', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    worker = WorkerClient([args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
                           '--capacity', '2048', '--max-sessions', '1', '--active-sequences', '1', '--draft-count', '0'])
    server = thread = connection = None
    try:
        server = ResponsesServer(('127.0.0.1', args.port), TextEngine(worker, FixtureCodec(worker.vocabulary), 'hyle'),
                                 args.metrics)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = socket.create_connection(('127.0.0.1', args.port), timeout=60)
        body = b'{"model":"hyle","input":"valid","max_output_tokens":1024,"stream":true}'
        connection.sendall(b'POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n'
                           + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
        data = b''
        while b'event: response.output_text.delta\n' not in data:
            part = connection.recv(65536)
            assert part, data
            data += part
        server.shutdown()
        thread.join(timeout=10)
        assert not thread.is_alive()
        metrics = server.engine.metrics.render().decode()
        assert 'hyle_requests_total 1\n' in metrics and 'hyle_requests_completed_total 0\n' in metrics, metrics
        server.server_close()
        metrics = server.engine.metrics.render().decode()
        assert 'hyle_requests_failed_total 1\n' in metrics and 'hyle_requests_completed_total 0\n' in metrics, metrics
        worker.close(timeout=60)
        print('active HTTP request cancelled; handler exited; worker acknowledged shutdown', flush=True)
    finally:
        if connection is not None:
            connection.close()
        if server is not None:
            if thread is not None and thread.is_alive():
                server.shutdown()
                thread.join()
            server.server_close()
        worker.close(timeout=60)


if __name__ == '__main__':
    main()
