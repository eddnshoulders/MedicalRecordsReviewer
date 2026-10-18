import io
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from reviewer import app as reviewer_app, core

class AdapterTests(unittest.TestCase):
    @patch('reviewer.core.urllib.request.build_opener')
    def test_model_list_uses_local_inventory_and_preserves_types(self, build):
        build.return_value.open.return_value=io.BytesIO(json.dumps({'models':[
            {'key':'local-chat-model','type':'llm'},
            {'key':'local-embedding-model','type':'embedding'},
        ]}).encode())
        self.assertEqual(core.list_models(),['local-chat-model','local-embedding-model'])
        details=build.return_value.open.call_args.args[0]
        self.assertEqual(details.full_url,'http://127.0.0.1:1234/api/v1/models')

    @patch('reviewer.core.list_model_details',return_value=[
        {'key':'local-chat-model','type':'llm'},
        {'key':'local-embedding-model','type':'embedding'},
    ])
    def test_model_api_exposes_types_for_embedding_picker(self, _details):
        result=reviewer_app.models()
        self.assertEqual(result['types']['local-embedding-model'],'embedding')

    @patch('reviewer.core.urllib.request.build_opener')
    @patch('reviewer.core.list_model_details')
    def test_medgemma_context_is_explicitly_loaded_at_16k(self, details, build):
        details.return_value=[{'key':'medgemma-1.5-4b-it','loaded_instances':[
            {'id':'medgemma-1.5-4b-it','config':{'context_length':8192}}]}]
        build.return_value.open.side_effect=[
            io.BytesIO(b'{"instance_id":"medgemma-1.5-4b-it"}'),
            io.BytesIO(b'{"load_config":{"context_length":16384}}'),
        ]
        self.assertEqual(core.ensure_model_context('medgemma-1.5-4b-it',16384),16384)
        unload_request,load_request=build.return_value.open.call_args_list[0][0][0],build.return_value.open.call_args_list[1][0][0]
        self.assertEqual(unload_request.full_url,'http://127.0.0.1:1234/api/v1/models/unload')
        self.assertEqual(load_request.full_url,'http://127.0.0.1:1234/api/v1/models/load')
        self.assertEqual(json.loads(load_request.data),{'model':'medgemma-1.5-4b-it','context_length':16384,'echo_load_config':True})

    @patch('reviewer.core.urllib.request.build_opener')
    @patch('reviewer.core.list_model_details')
    def test_medgemma_context_is_not_reloaded_when_already_16k(self, details, build):
        details.return_value=[{'key':'medgemma-1.5-4b-it','loaded_instances':[
            {'id':'medgemma-1.5-4b-it','config':{'context_length':16384}}]}]
        self.assertEqual(core.ensure_model_context('medgemma-1.5-4b-it',16384),16384)
        build.assert_not_called()

    @patch('reviewer.core.list_model_contexts')
    def test_stage_budgets_use_loaded_context_and_output_caps(self, contexts):
        contexts.return_value={
            'nuextract':{'loaded':True,'context_length':32768,'max_context_length':131072},
            'medgemma':{'loaded':True,'context_length':16384,'max_context_length':131072},
        }
        result=reviewer_app.model_contexts('lighton','nuextract','medgemma')['stages']
        self.assertEqual((result['ocr']['max_input_tokens'],result['ocr']['max_output_tokens']),(8192,8192))
        self.assertEqual((result['evidence']['max_input_tokens'],result['evidence']['max_output_tokens']),(20768,12000))
        self.assertEqual((result['medgemma']['max_input_tokens'],result['medgemma']['max_output_tokens']),(14336,2048))
        self.assertFalse(result['medgemma']['output_limited_by_context'])

    @patch('reviewer.core.urllib.request.build_opener')
    def test_ocr_sends_image_without_evidence_schema(self, build):
        build.return_value.open.return_value=io.BytesIO(json.dumps({'output':[{'type':'reasoning','content':'must not be included'},{'type':'message','content':'Transcribed page'}],'stats':{'input_tokens':1200,'total_output_tokens':450}}).encode())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'page.png';path.write_bytes(b'synthetic-image')
            result=core.ocr_call('lighton',path)
            self.assertEqual(result['content'],'Transcribed page')
            self.assertEqual((result['input_tokens'],result['output_tokens']),(1200,450))
        request=build.return_value.open.call_args.args[0]
        body=json.loads(request.data)
        self.assertTrue(body['input'][0]['data_url'].startswith('data:image/png;base64,'))
        self.assertEqual(body['reasoning'],'off')
        self.assertEqual(body['store'],False)
        self.assertEqual(len(body['input']),1)
        self.assertNotIn('response_format',body)

    @patch('reviewer.core.urllib.request.build_opener')
    def test_schema_request_and_response(self, build):
        build.return_value.open.return_value = io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"events":[]}'}}],'usage':{'prompt_tokens':123,'completion_tokens':4}}).encode())
        result=core.model_call('local-model','synthetic text')
        self.assertEqual(result['content'],'{"events":[]}')
        self.assertEqual(result['input_tokens'],123)
        self.assertEqual(result['output_tokens'],4)
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url,'http://127.0.0.1:1234/v1/chat/completions')
        body=json.loads(request.data)
        self.assertEqual(body['response_format']['type'],'json_schema')
        self.assertEqual(body['model'],'local-model')
        self.assertEqual(body['max_tokens'],12000)
        self.assertEqual(body['temperature'],0)
        self.assertNotIn('top_p',body)
        self.assertNotIn('frequency_penalty',body)
        self.assertNotIn('repeat_penalty',body)

    @patch('reviewer.core.urllib.request.build_opener')
    def test_medgemma_request_uses_requested_sampling_and_output_settings(self, build):
        build.return_value.open.return_value=io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"events":[]}'}}],'usage':{'prompt_tokens':123,'completion_tokens':4}}).encode())
        core.model_call('medgemma-1.5-4b-it','synthetic text')
        request=build.return_value.open.call_args.args[0]
        body=json.loads(request.data)
        self.assertEqual(body['max_tokens'],2048)
        self.assertEqual(body['temperature'],0.75)
        self.assertEqual(body['top_p'],0.95)
        self.assertEqual(body['frequency_penalty'],0.2)
        self.assertEqual(body['repeat_penalty'],1.15)
        self.assertNotIn('do_sample',body)
        self.assertEqual(body['response_format']['type'],'json_schema')

    @patch('reviewer.core.urllib.request.build_opener')
    def test_truncation_rejected(self, build):
        build.return_value.open.return_value = io.BytesIO(json.dumps({'choices':[{'finish_reason':'length','message':{'content':'partial output'}}],'usage':{'prompt_tokens':468,'completion_tokens':12000},'model_info':{'context_length':32768}}).encode())
        with self.assertRaisesRegex(core.OutputLimitError,'token limit') as caught:
            core.model_call('local-model','synthetic text')
        self.assertEqual((caught.exception.content,caught.exception.input_tokens,caught.exception.output_tokens,caught.exception.finish_reason,caught.exception.context_length),('partial output',468,12000,'length',32768))
