"""All three text protocols against one existing local native worker; no build/download."""
import argparse
import threading

from engine.text.app import TextApiServer
from engine.text.codec import HuggingFaceTextCodec
from engine.text.configuration import SessionPolicy
from engine.text.runtime import TextEngine
from engine.text.worker_client import WorkerClient


def main():
    import anthropic, httpx, openai
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=2048)
    parser.add_argument('--metrics', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    worker = WorkerClient([args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
        '--capacity', '2048', '--max-sessions', '2', '--active-sequences', '2', '--draft-count', '0'])
    try:
        assert worker.source_identity == (Path(args.inputs) / 'weights.tsv').read_text().splitlines()[1]
        codec = HuggingFaceTextCodec(args.tokenizer_dir, expected_vocabulary=worker.vocabulary)
        codec.validate_vocabulary(worker.vocabulary)
        engine = TextEngine(worker, codec, 'hyle', session_policy=SessionPolicy(kv_capacity=2))
        server = TextApiServer(('127.0.0.1', args.port), engine, args.metrics)
        thread = threading.Thread(target=server.serve_forever); thread.start()
        try:
            base = f'http://127.0.0.1:{args.port}'
            with openai.OpenAI(base_url=base + '/v1', api_key='local-test', max_retries=0,
                    http_client=httpx.Client(trust_env=False), _strict_response_validation=True) as oa, \
                 anthropic.Anthropic(base_url=base, api_key='local-test', max_retries=0,
                    http_client=httpx.Client(trust_env=False), _strict_response_validation=True) as ant:
                prompt = 'Reply with one English greeting.'
                messages = [{'role': 'user', 'content': prompt}]
                responses = oa.responses.create(model='hyle', input=prompt, max_output_tokens=8, store=False)
                chat = oa.chat.completions.create(model='hyle', messages=messages, max_completion_tokens=8)
                message = ant.messages.create(model='hyle', messages=messages, max_tokens=8)
                expected = responses.output_text
                assert expected == chat.choices[0].message.content == message.content[0].text
                assert responses.usage.output_tokens == chat.usage.completion_tokens == message.usage.output_tokens
                chunks = list(oa.chat.completions.create(model='hyle', messages=messages, max_tokens=8,
                    stream=True, stream_options={'include_usage': True}))
                assert ''.join(choice.delta.content or '' for chunk in chunks for choice in chunk.choices) == expected
                assert chunks[-1].usage.completion_tokens == chat.usage.completion_tokens
                with ant.messages.stream(model='hyle', messages=messages, max_tokens=8) as stream:
                    assert ''.join(stream.text_stream) == expected
                    assert stream.get_final_message().usage.output_tokens == message.usage.output_tokens
                print('PASS local CUDA: Responses, Chat and Messages text/usage; strict SDK stream helpers', flush=True)
                prefilled = ant.messages.create(model='hyle', messages=[*messages, {'role': 'assistant', 'content': 'Hello'}], max_tokens=4)
                assert prefilled.usage.output_tokens > 0 and prefilled.stop_reason in ('end_turn', 'max_tokens')
                print('PASS local tokenizer/native assistant prefill', flush=True)
                if expected:
                    stopped = oa.chat.completions.create(model='hyle', messages=messages, max_tokens=8, stop=expected[0])
                    assert stopped.choices[0].message.content == '' and stopped.choices[0].finish_reason == 'stop'
                    assert worker.healthy
                    assert oa.chat.completions.create(model='hyle', messages=messages, max_tokens=8).choices[0].message.content == expected
                    print('PASS local native stop cancellation/retirement and subsequent request', flush=True)
                print('SDK versions', openai.__version__, anthropic.__version__, 'tokens', message.usage.output_tokens, flush=True)
        finally:
            server.shutdown(); thread.join(timeout=3); server.server_close()
    finally: worker.close(timeout=120)


if __name__ == '__main__': main()
