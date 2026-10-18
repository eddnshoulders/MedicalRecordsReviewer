import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf as fitz
from reviewer import core


class SemanticSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.previous=core.ROOT
        core.ROOT=Path(self.temp.name)/'store'
        core.init()
        self.pdf=Path(self.temp.name)/'records.pdf'
        doc=fitz.open()
        doc.new_page().insert_text((50,50),'Increased hepatic stiffness suggested liver scarring.')
        doc.new_page().insert_text((50,50),'Employment salary and work history were recorded.')
        doc.save(self.pdf)
        doc.close()

    def tearDown(self):
        core.ROOT=self.previous
        self.temp.cleanup()

    def test_chunk_text_overlaps_and_covers_input(self):
        text=' '.join(f'word{i}' for i in range(500))
        chunks=core.chunk_text(text,max_chars=120,overlap=20)
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(len(chunk)<=120 for chunk in chunks))
        self.assertIn(chunks[0][-12:],chunks[1])
        self.assertIn('word499',chunks[-1])

    @patch('reviewer.core.urllib.request.build_opener')
    def test_embedding_adapter_sends_local_request_and_orders_results(self, build):
        payload={'data':[{'index':1,'embedding':[0.0,1.0]},{'index':0,'embedding':[1.0,0.0]}]}
        build.return_value.open.return_value=io.BytesIO(json.dumps(payload).encode())
        self.assertEqual(core.embedding_call('local-embed',['first','second']),[[1.0,0.0],[0.0,1.0]])
        request=build.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url,'http://127.0.0.1:1234/v1/embeddings')
        self.assertEqual(json.loads(request.data)['input'],['first','second'])

    def test_index_semantic_chunks_and_rank_related_pages(self):
        document_id=core.ingest(self.pdf)
        texts={1:'Increased hepatic stiffness suggested liver scarring.',2:'Employment salary and work history were recorded.'}
        with core.connect() as db:
            pages=db.execute('SELECT id,number FROM pages WHERE document_id=? ORDER BY number',(document_id,)).fetchall()
            for page in pages:
                db.execute("INSERT INTO transcriptions(page_id,model,text,status,created) VALUES(?,'mock',?,'complete',?)",(page['id'],texts[page['number']],core.now()))
            job_id=db.execute("INSERT INTO semantic_jobs(status,model,created,updated,total) VALUES('queued','local-embed',?,?,2)",(core.now(),core.now())).lastrowid

        def fake_embeddings(model, inputs):
            return [[1.0,0.0] if ('liver' in value.lower() or 'hepatic' in value.lower() or 'scarring' in value.lower()) else [0.0,1.0] for value in inputs]

        result=core.build_semantic_index(job_id,fake_embeddings)
        self.assertEqual((result['status'],result['completed'],result['total']),('complete',2,2))
        with core.connect() as db:
            page=db.execute('SELECT id FROM pages WHERE document_id=? AND number=1',(document_id,)).fetchone()
            evidence_job=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'complete','mock','now','lmstudio','evidence')",(document_id,)).lastrowid
            run_id=db.execute("INSERT INTO runs(job_id,page_id,model,output,status,created) VALUES(?,?,'mock','{}','complete','now')",(evidence_job,page['id'])).lastrowid
            for indexed_page in db.execute('SELECT id FROM pages WHERE document_id=?',(document_id,)).fetchall():
                db.execute("INSERT INTO evidence(run_id,page_id,payload,quote,status) VALUES(?,?, '{}','source','unreviewed')",(run_id,indexed_page['id']))
        found=core.semantic_search('evidence of liver scarring',limit=100,call=fake_embeddings)
        self.assertEqual(found[0]['number'],1)
        self.assertGreater(found[0]['semantic_score'],found[1]['semantic_score'])
        self.assertIn('hepatic stiffness',found[0]['match_text'])
        core.delete_document(document_id)
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM page_embeddings').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT count(*) FROM semantic_page_index').fetchone()[0],0)


if __name__=='__main__':
    unittest.main()
