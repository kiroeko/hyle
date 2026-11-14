"""Model discovery uses prepared facts and both official SDK wire formats."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import threading
import unittest
import urllib.error
import urllib.request

from engine.text.app import TextApiServer
from engine.text.codec import HuggingFaceTextCodec
from engine.text.configuration import ExecutionChoice,PreparedChoices,SamplingChoice
from engine.text.contracts import ExecutionProfileFacts
from engine.text.runtime import TextEngine
from engine.models.qwen35.tool_format import ToolFormat


def engine():
    choices=PreparedChoices(executions=(ExecutionChoice(drafts=3,assets=(('vision.encoder',Path('private-input.asset')),)),
        ExecutionChoice('pictures')),samplers=(SamplingChoice(),SamplingChoice('chat',True,.8,40,.95)))
    worker=SimpleNamespace(healthy=True,context=128,vocabulary=42,source_identity='actual-source',stop_ids=frozenset(),
        execution_profiles=(ExecutionProfileFacts('default','ordinary',0,('text',),(0,1)),
                            ExecutionProfileFacts('pictures','another-model-proposal',2,('text','image'),(0,1))))
    codec=SimpleNamespace(stop_ids=frozenset())
    return TextEngine(worker,codec,'hyle',choices=choices)


class ModelFactsTest(unittest.TestCase):
    def test_declared_assets_and_draft_preferences_never_claim_native_capabilities(self):
        value=engine().describe_capabilities()
        self.assertEqual((value['context_tokens'],value['vocabulary_size'],value['source_identity']),(128,42,'actual-source'))
        profile=value['execution_profiles'][0]
        self.assertEqual((profile['algorithm'],profile['max_draft_tokens'],profile['input_modalities']),('ordinary',0,['text']))
        self.assertEqual(profile['co_resident_with'],['default','pictures'])
        self.assertNotIn('private-input.asset',json.dumps(value))
        self.assertEqual(value['sampling_presets'][0]['temperature'],0.)
        self.assertEqual(value['sampling_presets'][1]['top_k'],40)
        self.assertEqual(value['structured_output'],{'json_mode':False,'strict_schema':False})

    def test_codec_reports_its_real_optional_recipe_and_returns_owned_descriptions(self):
        codec=object.__new__(HuggingFaceTextCodec);codec.image_processor=None;codec.model_format=None
        self.assertFalse(codec.describe_capabilities()['tools']['supported'])
        codec.model_format=ToolFormat()
        first=codec.describe_capabilities()
        self.assertTrue(first['tools']['supported']);self.assertFalse(first['tools']['strict'])
        first['tools']['choices'].clear()
        self.assertIn('auto',codec.describe_capabilities()['tools']['choices'])
        self.assertEqual(codec.describe_capabilities()['input_modalities'],['text'])
        codec.image_processor=object()
        self.assertEqual(codec.describe_capabilities()['input_modalities'],['text','image'])

    def test_unknown_legacy_executor_facts_are_not_invented(self):
        value=engine();value.execution_profiles=()
        self.assertEqual(value.describe_capabilities()['execution_profiles'],[])
        value.worker.healthy=False
        self.assertFalse(value.describe_capabilities()['available'])


class ModelHttpTest(unittest.TestCase):
    def setUp(self):
        self.engine=engine();self.server=TextApiServer(('127.0.0.1',0),self.engine,True)
        self.thread=threading.Thread(target=self.server.serve_forever);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown();self.thread.join(timeout=3);self.server.server_close()
    def get(self,path,anthropic=False):
        request=urllib.request.Request(self.url+path,headers={'anthropic-version':'2023-06-01'} if anthropic else {})
        try:response=urllib.request.urlopen(request,timeout=3)
        except urllib.error.HTTPError as error:response=error
        with response:return response.status,json.load(response)
    def test_both_formats_and_cursor_boundaries_share_actual_facts(self):
        status,oa=self.get('/v1/models');self.assertEqual(status,200)
        status,ant=self.get('/v1/models',True);self.assertEqual(status,200)
        self.assertEqual(oa['data'][0]['hyle'],ant['data'][0]['hyle'])
        self.assertEqual(oa['object'],'list');self.assertNotIn('object',ant)
        self.assertEqual(ant['data'][0]['type'],'model');self.assertNotIn('type',oa['data'][0])
        self.assertEqual(self.get('/v1/models/hyle')[1],oa['data'][0])
        self.assertEqual(self.get('/v1/models/hyle',True)[1],ant['data'][0])
        for direction in ('after_id','before_id'):
            status,value=self.get('/v1/models?'+direction+'=hyle&limit=1',True)
            self.assertEqual((status,value['data'],value['has_more']),(200,[],False))
    def test_invalid_queries_and_model_names_do_not_run_inference(self):
        for query in ('limit=0','limit=1001','limit=x','limit=1&limit=2','after_id=missing','after_id=hyle&before_id=hyle','unknown=x'):
            self.assertEqual(self.get('/v1/models?'+query,True)[0],400)
        for ant in (False,True):
            status,value=self.get('/v1/models/missing',ant)
            self.assertEqual(status,404);self.assertEqual(value['error']['type'],'not_found_error')
        self.assertTrue(self.engine.healthy)


SDK_AVAILABLE=all(importlib.util.find_spec(name) for name in ('openai','anthropic','httpx'))


@unittest.skipUnless(SDK_AVAILABLE,'official SDKs are optional')
class ModelSdkTest(ModelHttpTest):
    def test_official_sdk_list_retrieve_empty_pages_and_not_found(self):
        import anthropic,httpx,openai
        with openai.OpenAI(base_url=self.url+'/v1',api_key='local-test',max_retries=0,
                http_client=httpx.Client(trust_env=False),_strict_response_validation=True) as oa:
            page=oa.models.list();self.assertEqual([model.id for model in page.data],['hyle'])
            self.assertEqual(oa.models.retrieve('hyle').model_dump()['hyle']['context_tokens'],128)
            with self.assertRaises(openai.NotFoundError):oa.models.retrieve('missing')
        with anthropic.Anthropic(base_url=self.url,api_key='local-test',max_retries=0,
                http_client=httpx.Client(trust_env=False),_strict_response_validation=True) as ant:
            page=ant.models.list(limit=1);self.assertFalse(page.has_next_page())
            self.assertEqual(page.data[0].display_name,'hyle')
            self.assertEqual(ant.models.retrieve('hyle').max_input_tokens,128)
            self.assertEqual(ant.models.list(after_id='hyle').data,[])
            with self.assertRaises(anthropic.NotFoundError):ant.models.retrieve('missing')


if __name__=='__main__':unittest.main()
