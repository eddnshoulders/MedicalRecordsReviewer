import json
import tempfile
import unittest
from pathlib import Path

import pymupdf as fitz
from fastapi import HTTPException

from reviewer import app, core


class ReasoningToolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.previous=core.ROOT
        core.ROOT=Path(self.temp.name)/'store'
        core.init()
        pdf=fitz.open()
        pdf.new_page().insert_text((50,50),'Treatment began in 2003 and caused fatigue.')
        pdf.save(Path(self.temp.name)/'source.pdf')
        pdf.close()
        self.document_id=core.ingest(Path(self.temp.name)/'source.pdf')
        with core.connect() as db:
            page=db.execute('SELECT id FROM pages WHERE document_id=?',(self.document_id,)).fetchone()
            transcription_id=db.execute("INSERT INTO transcriptions(page_id,model,text,status,created) VALUES(?,'mock-ocr','Treatment began in 2003 and caused fatigue.','complete',?)",(page['id'],core.now())).lastrowid
            job_id=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'complete','mock-extractor',?,'lmstudio','evidence')",(self.document_id,core.now())).lastrowid
            run_id=db.execute("INSERT INTO runs(job_id,page_id,model,output,status,created,transcription_id) VALUES(?,?,?,'{}','complete',?,?)",(job_id,page['id'],'mock-extractor',core.now(),transcription_id)).lastrowid
            payload={'event_type':'symptom','treatment':'Fatigue during treatment','date_as_written':'2003','date_basis':'explicit','source_quote':'Treatment began in 2003 and caused fatigue.','quote_match':True}
            db.execute('INSERT INTO evidence(run_id,page_id,payload,quote,start,end,status,quote_match) VALUES(?,?,?,?,?,?,?,1)',(run_id,page['id'],json.dumps(payload),payload['source_quote'],0,len(payload['source_quote']),'confirmed'))

    def tearDown(self):
        core.ROOT=self.previous
        self.temp.cleanup()

    def test_get_source_returns_latest_ocr_and_provenance(self):
        pages=core.get_source(self.document_id,[1])
        self.assertEqual(len(pages),1)
        self.assertEqual(pages[0]['source_text'],'Treatment began in 2003 and caused fatigue.')
        self.assertEqual(pages[0]['ocr_model'],'mock-ocr')
        self.assertEqual(pages[0]['text_source'],'ocr')

    def test_document_page_map_provides_stable_page_ids(self):
        result=core.get_document_page_map(self.document_id)
        self.assertEqual(result['pages'][0]['page'],1)
        self.assertEqual(result['pages'][0]['page_id'],f"{self.document_id}-1")

    def test_timeline_is_sorted_and_keeps_candidate_provenance(self):
        rows=core.get_timeline()
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['date_as_written'],'2003')
        self.assertEqual(rows[0]['review_status'],'confirmed')
        self.assertEqual(rows[0]['page'],1)
        self.assertEqual(rows[0]['source_quote'],'Treatment began in 2003 and caused fatigue.')

    def test_assessment_criteria_never_invents_unconfigured_rules(self):
        result=core.get_assessment_criteria('treatment')
        self.assertFalse(result['configured'])
        self.assertEqual(result['criteria'],[])
        self.assertIn('Do not infer or invent',result['message'])
        with core.connect() as db:
            db.execute('INSERT INTO assessment_criteria(topic,title,body,source_title,source_url,version,created) VALUES(?,?,?,?,?,?,?)',('treatment','Treatment duration','Authoritative rule text','Official source','https://example.invalid/rules','v1',core.now()))
        result=core.get_assessment_criteria('treatment')
        self.assertTrue(result['configured'])
        self.assertEqual(result['criteria'][0]['version'],'v1')

    def test_search_tool_rejects_unknown_filters(self):
        request=app.EvidenceSearchRequest(query='fatigue',filters={'not_a_filter':'anything'})
        with self.assertRaises(HTTPException) as raised:
            app.tool_search_evidence(request)
        self.assertEqual(raised.exception.status_code,400)


if __name__=='__main__':
    unittest.main()
