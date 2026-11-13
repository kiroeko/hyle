"""Image wire normalization, model preparation, patch layout and session isolation."""
import base64
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from engine.models.qwen35.vision_preparation import prepare_vision, vision_shapes
from engine.models.qwen35.vision_processing import ImageProcessor
from engine.text.media import ImageInput, PromptAttachment, content_parts, inline_image
from engine.text.worker_client import _prompt_frame
from engine.text.chat_completions import parse_chat_completion
from engine.text.anthropic import parse_messages
from engine.text.responses import parse_responses_input
from engine.text.configuration import Selection, SessionPolicy
from engine.text.sessions import TextSessions
from tests.tooling.text_sessions_test import Worker, complete


def png(color='red', size=(8,8)):
    stream=BytesIO();Image.new('RGB',size,color).save(stream,format='PNG');return stream.getvalue()


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        ids=[1]
        for message in messages:
            for part in message['content']:
                ids.extend([98,99,97] if part['type']=='image' else [2])
        return ids+[3]


class VisionPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'.scratch')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.vision=dict(depth=1,hidden_size=16,intermediate_size=32,num_heads=4,patch_size=2,
            temporal_patch_size=2,spatial_merge_size=2,num_position_embeddings=16,out_hidden_size=8,
            in_channels=3,hidden_act='gelu_pytorch_tanh',deepstack_visual_indexes=[])
        config=dict(model_type='qwen3_5',vision_config=self.vision,image_token_id=99,vision_start_token_id=98,
            vision_end_token_id=97,text_config=dict(hidden_size=8,head_dim=64,rope_parameters=dict(
                mrope_interleaved=True,mrope_section=[3,3,2],partial_rotary_factor=.25)))
        (self.root/'config.json').write_text(json.dumps(config))
        (self.root/'preprocessor_config.json').write_text(json.dumps(dict(patch_size=2,merge_size=2,
            temporal_patch_size=2,image_mean=[.5]*3,image_std=[.5]*3,size=dict(longest_edge=1024,shortest_edge=16))))
        self.processor=ImageProcessor(self.root,Tokenizer(),context=64)
        self.url='data:image/png;base64,'+base64.b64encode(png()).decode()

    def test_protocols_preserve_order_and_share_image_bytes(self):
        text={'type':'text','text':'question'}
        chat=content_parts([text,{'type':'image_url','image_url':{'url':self.url}}],protocol='chat',role='user')
        response=content_parts([{'type':'input_text','text':'question'},{'type':'input_image','image_url':self.url}],protocol='responses',role='user')
        anthropic=content_parts([text,{'type':'image','source':{'type':'base64','media_type':'image/png','data':self.url.split(',')[1]}}],protocol='anthropic',role='user')
        self.assertEqual(chat,response);self.assertEqual(chat,anthropic)
        self.assertEqual(chat[1]['image'].data,png())

    def test_three_public_parsers_accept_images(self):
        chat,_=parse_chat_completion({'model':'hyle','messages':[{'role':'user','content':[
            {'type':'image_url','image_url':{'url':self.url}}]}]},'hyle')
        anthropic,_=parse_messages({'model':'hyle','max_tokens':8,'messages':[{'role':'user','content':[
            {'type':'image','source':{'type':'base64','media_type':'image/png','data':self.url.split(',')[1]}}]}]},'hyle')
        response=parse_responses_input({'model':'hyle','max_output_tokens':8,'input':[{'role':'user','content':[
            {'type':'input_image','image_url':self.url}]}]},'hyle')[0]
        self.assertEqual(chat.messages,anthropic.messages);self.assertEqual(response,chat.messages)

    def test_remote_file_and_malformed_sources_are_rejected(self):
        for url in ('https://example.com/image.png','file:///private.png','data:image/png,abc',
                    'data:text/plain;base64,YQ==','data:image/png;base64,@@@','data:image/png;base64,'):
            with self.subTest(url=url),self.assertRaises(ValueError):inline_image(url)

    def test_images_cannot_hide_in_system_or_assistant_messages(self):
        for role in ('system','assistant','developer'):
            with self.subTest(role=role),self.assertRaises(ValueError):
                content_parts([{'type':'input_image','image_url':self.url}],protocol='responses',role=role)

    def test_plain_text_normalization_stays_a_string(self):
        self.assertEqual(content_parts([{'type':'text','text':'a'},{'type':'text','text':'b'}],protocol='chat',role='user'),'ab')

    def test_patch_order_temporal_repeat_and_normalization(self):
        data=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
        stream=BytesIO();Image.fromarray(data).save(stream,format='PNG')
        grid,payload=self.processor.patches(ImageInput('image/png',stream.getvalue()))
        self.assertEqual(grid,(1,4,4))
        actual=np.frombuffer(payload,dtype='<f4').reshape(16,3,2,2,2)
        normalized=(data.astype(np.float32)*np.float32(1/255)-np.float32(.5))/np.float32(.5)
        coords=[(by+dy,bx+dx) for by in range(0,4,2) for bx in range(0,4,2) for dy in range(2) for dx in range(2)]
        for i,(y,x) in enumerate(coords):
            patch=normalized[y*2:y*2+2,x*2:x*2+2].transpose(2,0,1)
            np.testing.assert_array_equal(actual[i,:,0],patch);np.testing.assert_array_equal(actual[i,:,1],patch)

    def test_declared_mime_and_animation_are_checked(self):
        with self.assertRaises(ValueError):self.processor.patches(ImageInput('image/jpeg',png()))
        stream=BytesIO();Image.new('RGB',(8,8),'red').save(stream,format='GIF',save_all=True,append_images=[Image.new('RGB',(8,8),'blue')])
        with self.assertRaises(ValueError):self.processor.patches(ImageInput('image/gif',stream.getvalue()))

    def test_multiple_images_expand_only_their_exact_placeholder_intervals(self):
        messages=[{'role':'user','content':[{'type':'image','image':ImageInput('image/png',png())},
            {'type':'text','text':'between'},{'type':'image','image':ImageInput('image/png',png('blue'))}]}]
        result=self.processor.prepare(messages,max_tokens=32)
        self.assertEqual(len(result.attachments),2)
        for part in result.attachments:
            self.assertEqual(result.ids[part.begin-1],98);self.assertEqual(result.ids[part.begin+part.rows],97)
            self.assertEqual(result.ids[part.begin:part.begin+part.rows],[99]*part.rows)
        self.assertNotEqual(result.attachments[0].payload,result.attachments[1].payload)
        self.assertLessEqual(len(result.ids),32)

    def test_template_cannot_drop_images(self):
        self.processor.tokenizer=mock.Mock();self.processor.tokenizer.apply_chat_template.return_value=[1,2,3]
        with self.assertRaises(ValueError):self.processor.prepare([{'role':'user','content':[{'type':'image','image':ImageInput('image/png',png())}]}])

    def test_image_context_budget_and_low_detail(self):
        image=ImageInput('image/png',png(size=(128,128)),detail='low')
        grid,_=self.processor.patches(image,max_rows=64)
        self.assertEqual(grid[1]*grid[2]//4,1)
        with self.assertRaises(ValueError):self.processor.prepare([{'role':'user','content':[{'type':'image','image':image}]}],max_tokens=3)

    def test_transport_is_plain_for_text_and_checks_conditioning_extents(self):
        part=PromptAttachment('model.image.v1',1,1,(1,2,2),b'\0\1\2')
        self.assertEqual(_prompt_frame([1,2]),'1,2')
        self.assertEqual(_prompt_frame([1,2],[part]),'1,2;model.image.v1:1:1:1,2,2:AAEC')
        for invalid in (replace(part,begin=2),replace(part,rows=-1),replace(part,shape=(0,)),replace(part,format='bad;type')):
            with self.assertRaises(ValueError):_prompt_frame([1,2],[invalid])
        with self.assertRaises(ValueError):_prompt_frame([1,2],[part,part])
        self.assertIn(':2:0:',_prompt_frame([1,2],[replace(part,begin=2,rows=0)]))

    def test_asset_reuses_original_ranges_and_has_stable_source_binding(self):
        header,payload,offset={},bytearray(),0
        for name,shape in vision_shapes(self.vision).items():
            size=int(np.prod(shape))*4
            source_shape=shape if not name.endswith('patch_embed.proj.weight') else [16,3,2,2,2]
            header[name]=dict(dtype='F32',shape=source_shape,data_offsets=[offset,offset+size])
            payload.extend(bytes(size));offset+=size
        raw=json.dumps(header).encode()
        (self.root/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+payload)
        output=self.root/'vision.tsv'
        with mock.patch('engine.model_preparation.prepared_source_identity',return_value=('a'*64,100)):
            first=prepare_vision(self.root,self.root,output)
            second=prepare_vision(self.root,self.root,output)
        self.assertEqual(first,second)
        lines=output.read_text().splitlines()
        self.assertEqual(lines[1],'a'*64)
        self.assertEqual(len(lines)-6,21)
        self.assertTrue(all('model.safetensors' in row for row in lines[6:]))

    def test_incomplete_visual_checkpoint_is_an_error(self):
        raw=b'{}';(self.root/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw)
        with mock.patch('engine.model_preparation.prepared_source_identity',return_value=('a'*64,100)):
            with self.assertRaisesRegex(ValueError,'missing visual tensors'):
                prepare_vision(self.root,self.root,self.root/'vision.tsv')

    def test_same_placeholder_ids_cannot_reuse_different_image_state(self):
        worker=Worker();owner=TextSessions(worker,SessionPolicy(kv_capacity=1));self.addCleanup(owner.close)
        a=PromptAttachment('model.image.v1',1,1,(1,2,2),b'a')
        first=owner.start('a',[],[1,2],1,Selection(),attachments=(a,));complete(first,'a')
        parent=owner.parent('a')
        changed=owner.start('b',[],[1,2,3,4],1,Selection(),parent=parent,attachments=(replace(a,payload=b'b'),))
        self.assertEqual(changed.cached_tokens,0);self.assertEqual(len(worker.requests),2)
        complete(changed,'b')

    def test_identical_image_identity_continues_complete_session_state(self):
        worker=Worker();owner=TextSessions(worker,SessionPolicy(kv_capacity=1));self.addCleanup(owner.close)
        a=PromptAttachment('model.image.v1',1,1,(1,2,2),b'a')
        first=owner.start('a',[],[1,2],1,Selection(),attachments=(a,));complete(first,'a')
        continued=owner.start('b',[],[1,2,3,4],1,Selection(),parent=owner.parent('a'),attachments=(a,))
        self.assertEqual(continued.cached_tokens,2);self.assertEqual(len(worker.requests),1)
        complete(continued,'b')

    def test_saved_media_history_is_independent_of_ingress_containers(self):
        worker=Worker();owner=TextSessions(worker,SessionPolicy(kv_capacity=1));self.addCleanup(owner.close)
        image=ImageInput('image/png',png())
        messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'color?'}]}]
        turn=owner.start('a',messages,[1,2],1,Selection())
        messages[0]['content'].clear()
        complete(turn,'a')
        saved=owner.parent('a').messages[0]['content']
        self.assertEqual(len(saved),2)
        self.assertEqual(saved[1]['text'],'color?')
        self.assertIs(saved[0]['image'].data,image.data)


if __name__=='__main__':unittest.main()
