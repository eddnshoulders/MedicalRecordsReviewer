import tempfile
import unittest
from pathlib import Path

import pymupdf as fitz
from reviewer import app, core


class UiPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.previous=core.ROOT
        core.ROOT=Path(self.temp.name)/'store'
        core.init()
        source=Path(self.temp.name)/'source.pdf'
        doc=fitz.open()
        doc.new_page().insert_text((50,50),'Native PDF text that should not be polled.')
        doc.save(source)
        doc.close()
        self.document_id=core.ingest(source)
        self.page_id=core.get_document_page_map(self.document_id)['pages'][0]['page_id']

    def tearDown(self):
        core.ROOT=self.previous
        self.temp.cleanup()

    def test_indexes_exist_and_init_is_repeatable(self):
        core.init()
        with core.connect() as db:
            names={r['name'] for r in db.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        self.assertTrue({name for name,_ in core.INDEXES}<=names)

    def test_page_status_queries_use_indexes(self):
        with core.connect() as db:
            plan=' '.join(r['detail'] for r in db.execute(
                "EXPLAIN QUERY PLAN SELECT id FROM transcriptions WHERE page_id=? AND status='complete' ORDER BY id DESC LIMIT 1",(self.page_id,)))
        self.assertIn('ix_transcriptions_page',plan)

    def test_version_changes_on_write_and_not_on_read(self):
        before=app.version()['version']
        app.state();app.dataset();app.pages(self.document_id)
        self.assertEqual(app.version()['version'],before)
        with core.connect() as db:
            db.execute("INSERT INTO transcriptions(page_id,model,text,status,created) VALUES(?,'mock','OCR text','complete',?)",(self.page_id,core.now()))
        self.assertGreater(app.version()['version'],before)

    def test_page_list_carries_status_not_text(self):
        with core.connect() as db:
            db.execute("INSERT INTO transcriptions(page_id,model,text,status,created) VALUES(?,'mock','OCR text','complete',?)",(self.page_id,core.now()))
        page=app.pages(self.document_id)[0]
        self.assertNotIn('text',page)
        self.assertNotIn('ocr_text',page)
        self.assertTrue(page['has_ocr'])
        self.assertIn('Native PDF text',app.page_native_text(self.page_id)['text'])

    def test_runs_include_output_only_for_failures(self):
        with core.connect() as db:
            job=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'needs_attention','mock',?,'lmstudio','evidence')",(self.document_id,core.now())).lastrowid
            db.execute("INSERT INTO runs(job_id,page_id,model,output,status,created) VALUES(?,?,'mock','large successful output','complete',?)",(job,self.page_id,core.now()))
            other=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'needs_attention','mock',?,'lmstudio','medgemma')",(self.document_id,core.now())).lastrowid
            db.execute("INSERT INTO runs(job_id,page_id,model,output,status,error,created) VALUES(?,?,'mock','partial','failed','cut off',?)",(other,self.page_id,core.now()))
        runs={run['status']:run for run in app.runs(self.document_id)}
        self.assertIsNone(runs['complete']['output'])
        self.assertEqual(runs['failed']['output'],'partial')

    def test_state_limits_history_but_keeps_open_file_and_active_jobs(self):
        with core.connect() as db:
            db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'complete','mock',?,'lmstudio','ocr')",(self.document_id,core.now()))
            old_id=db.execute('SELECT max(id) FROM jobs').fetchone()[0]
            running=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES('other','running','mock',?,'lmstudio','ocr')",(core.now(),)).lastrowid
            db.executemany("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES('other','complete','mock',?,'lmstudio','ocr')",
                           [(core.now(),)]*(app.STATE_JOB_LIMIT+5))
        ids={job['id'] for job in app.state()['jobs']}
        self.assertNotIn(old_id,ids)
        self.assertIn(running,ids)
        self.assertIn(old_id,{job['id'] for job in app.state(self.document_id)['jobs']})


if __name__=='__main__':
    unittest.main()
