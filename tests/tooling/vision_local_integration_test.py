"""Real local CUDA image understanding through all three HTTP protocols.

Requires existing Qwen3.5 inputs, tokenizer and prepared vision asset. Does not
build, download or modify model files. Images are small deterministic fixtures.
"""
import argparse
import base64
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import threading

from PIL import Image

from engine.model_preparation import local_image_processor, validate_local_tokenizer
from engine.text.app import TextApiServer
from engine.text.codec import HuggingFaceTextCodec
from engine.text.configuration import ExecutionChoice, PreparedChoices, SamplingChoice, SessionPolicy
from engine.text.media import PromptAttachment, ImageInput
from engine.text.runtime import TextEngine
from engine.text.worker_client import WorkerClient


def main():
    import anthropic, httpx, openai
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--tokenizer-dir',type=Path,required=True)
    parser.add_argument('--vision',type=Path,required=True)
    parser.add_argument('--port',type=int,default=2048)
    parser.add_argument('--metrics',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args()
    choices=PreparedChoices(executions=(ExecutionChoice(drafts=0,overlap=False,assets=(('vision.encoder',args.vision),)),),
        samplers=(SamplingChoice(),SamplingChoice('random',True,.8,40,.95)),prefix_entries=4)
    codec=HuggingFaceTextCodec(args.tokenizer_dir)
    identity=validate_local_tokenizer(args.inputs,codec.preparation_tokenizer)
    codec.image_processor=local_image_processor(args.inputs,args.tokenizer_dir,codec.preparation_tokenizer,context=512)
    worker=WorkerClient([args.engine,'--engine-worker','--inputs',str(args.inputs),'--backend','cuda',
        '--capacity','512','--max-sessions','2','--active-sequences','2','--prefill-chunk-tokens','32',
        '--draft-count','0','--overlap','off','--profile-asset','vision.encoder='+str(args.vision),
        '--additional-sampling','random,0.8,40,0.95','--prefix-cache-entries','4'])
    try:
        assert worker.source_identity==identity[0]
        engine=TextEngine(worker,codec,'hyle',choices=choices,session_policy=SessionPolicy(kv_capacity=2))
        server=TextApiServer(('127.0.0.1',args.port),engine,args.metrics)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            base=f'http://127.0.0.1:{args.port}'
            def image(color):
                buffer=BytesIO();Image.new('RGB',(256,256),color).save(buffer,format='PNG')
                return base64.b64encode(buffer.getvalue()).decode('ascii')
            red,blue=image('red'),image('blue')
            question='What color is this image? Answer with one English color word.'
            def chat_parts(data):return [{'type':'image_url','image_url':{'url':'data:image/png;base64,'+data}},
                                          {'type':'text','text':question}]
            def response_parts(data):return [{'type':'input_image','image_url':'data:image/png;base64,'+data},
                                              {'type':'input_text','text':question}]
            def anthropic_parts(data):return [{'type':'image','source':{'type':'base64','media_type':'image/png','data':data}},
                                               {'type':'text','text':question}]
            with openai.OpenAI(base_url=base+'/v1',api_key='local-test',max_retries=0,
                    http_client=httpx.Client(trust_env=False),_strict_response_validation=True) as oa, \
                 anthropic.Anthropic(base_url=base,api_key='local-test',max_retries=0,
                    http_client=httpx.Client(trust_env=False),_strict_response_validation=True) as ant:
                responses=oa.responses.create(model='hyle',input=[{'role':'user','content':response_parts(red)}],max_output_tokens=16,store=False)
                chat=oa.chat.completions.create(model='hyle',messages=[{'role':'user','content':chat_parts(red)}],max_tokens=16)
                messages=ant.messages.create(model='hyle',messages=[{'role':'user','content':anthropic_parts(red)}],max_tokens=16)
                expected=responses.output_text
                assert 'red' in expected.lower(),expected
                assert expected==chat.choices[0].message.content==messages.content[0].text
                assert responses.usage.input_tokens==chat.usage.prompt_tokens==messages.usage.input_tokens
                print('PASS CUDA: image understanding, identical text and token usage through all three APIs; prefill chunks cross image rows',flush=True)
                chunks=list(oa.chat.completions.create(model='hyle',messages=[{'role':'user','content':chat_parts(red)}],max_tokens=16,stream=True))
                assert ''.join(c.delta.content or '' for chunk in chunks for c in chunk.choices)==expected
                with ant.messages.stream(model='hyle',messages=[{'role':'user','content':anthropic_parts(red)}],max_tokens=16) as stream:
                    assert ''.join(stream.text_stream)==expected
                streamed=list(oa.responses.create(model='hyle',input=[{'role':'user','content':response_parts(red)}],max_output_tokens=16,store=False,stream=True))
                assert ''.join(event.delta for event in streamed if event.type=='response.output_text.delta')==expected
                print('PASS all three official SDK image stream helpers',flush=True)
                saved=oa.responses.create(model='hyle',input=[{'role':'user','content':response_parts(red)}],max_output_tokens=16)
                following=oa.responses.create(model='hyle',previous_response_id=saved.id,input='What was the color? Answer with one color word.',max_output_tokens=16)
                assert 'red' in following.output_text.lower(),following.output_text
                assert following.metadata['hyle.context'] in ('continued','replayed'),following.metadata
                oa.responses.delete(saved.id);oa.responses.delete(following.id)
                print('PASS image Responses history; context route',following.metadata['hyle.context'],flush=True)
                def prepared_image(data):
                    return engine.prepare([{'role':'user','content':[{'type':'image','image':ImageInput('image/png',base64.b64decode(data))},
                        {'type':'text','text':question}]}],16)
                def read_turn(request,live):
                    tokens=[]
                    while True:
                        event=request.read(timeout=60)
                        if event.kind=='token':tokens.append(event.token);request.acknowledge()
                        elif event.kind=='turn_end' and live:request.acknowledge();return tokens
                        elif event.kind=='completed' and not live:return tokens
                        else:raise AssertionError(event)
                rp,bp=prepared_image(red),prepared_image(blue)
                native=worker.open_session(rp.ids,16,attachments=rp.attachments)
                try:
                    first=read_turn(native,True)
                    newline=codec.preparation_tokenizer.encode('\n',add_special_tokens=False)
                    suffix=newline+bp.ids
                    parts=tuple(replace(part,begin=part.begin+len(newline)) for part in bp.attachments)
                    native.append(suffix,16,attachments=parts)
                    continued=read_turn(native,True)
                finally:native.retire(timeout=120);native.close()
                tape=rp.ids+first+suffix
                replay_parts=rp.attachments+tuple(replace(part,begin=part.begin+len(rp.ids)+len(first)) for part in parts)
                replay=worker.submit(tape,16,attachments=replay_parts)
                try:replayed=read_turn(replay,False)
                finally:replay.close()
                assert continued==replayed,(continued,replayed)
                assert 'blue' in codec.preparation_tokenizer.decode(continued).lower()
                print('PASS native image KV/recurrent continuation, second-turn image, pending and rotary offset equal complete replay',flush=True)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    replies=list(pool.map(lambda data:oa.chat.completions.create(model='hyle',
                        messages=[{'role':'user','content':chat_parts(data)}],max_tokens=16).choices[0].message.content,(red,blue)))
                assert 'red' in replies[0].lower() and 'blue' in replies[1].lower(),replies
                multiple=oa.chat.completions.create(model='hyle',messages=[{'role':'user','content':[
                    *chat_parts(red)[:1],*chat_parts(blue)[:1],{'type':'text','text':'What color is the second image? Answer with one color word.'}]}],max_tokens=16)
                assert 'blue' in multiple.choices[0].message.content.lower(),multiple
                print('PASS two concurrent image users and ordered multiple images; no token-placeholder cache alias',flush=True)
                split=Image.new('RGB',(384,256),'red');split.paste('blue',(192,0,384,256))
                buffer=BytesIO();split.save(buffer,format='PNG')
                spatial=chat_parts(base64.b64encode(buffer.getvalue()).decode('ascii'))
                spatial[-1]['text']='What color is the right half of this image? Answer with one English color word.'
                answer=oa.chat.completions.create(model='hyle',messages=[{'role':'user','content':spatial}],max_tokens=16)
                assert 'blue' in answer.choices[0].message.content.lower(),answer
                print('PASS non-square image grid and spatially distinct patch content',flush=True)
                random=oa.chat.completions.create(model='hyle',messages=[{'role':'user','content':chat_parts(red)}],max_tokens=4,
                    extra_body={'hyle':{'sampling_profile':'random','seed':42}})
                assert random.usage.completion_tokens>0
                try:
                    oa.chat.completions.create(model='hyle',messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,YQ=='}}]}],max_tokens=8)
                    raise AssertionError('malformed image accepted')
                except openai.BadRequestError:pass
                bad=worker.submit([1,2],1,attachments=(PromptAttachment('unsupported.image',0,1,(1,),b'a'),))
                try:
                    event=bad.read(timeout=60)
                    assert event.kind=='error' and event.error_kind=='invalid',event
                finally:bad.close()
                bad_live=worker.open_session(rp.ids,16,attachments=rp.attachments)
                try:
                    read_turn(bad_live,True)
                    bad_live.append([1,2],1,attachments=(PromptAttachment('unsupported.image',0,1,(1,),b'a'),))
                    event=bad_live.read(timeout=60)
                    assert event.kind=='error' and event.error_kind=='invalid',event
                finally:bad_live.retire(timeout=120);bad_live.close()
                assert oa.chat.completions.create(model='hyle',messages=[{'role':'user','content':chat_parts(blue)}],max_tokens=16).choices[0].message.content.lower().find('blue')>=0
                print('PASS prepared random sampler, malformed image/attachment rejection and subsequent image request',flush=True)
                print('SDK versions',openai.__version__,anthropic.__version__,'image prompt tokens',chat.usage.prompt_tokens,flush=True)
        finally:server.shutdown();thread.join(timeout=3);server.server_close()
    finally:worker.close(timeout=120)


if __name__=='__main__':main()
