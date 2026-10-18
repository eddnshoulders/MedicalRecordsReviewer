import json
import tempfile
import unittest
from pathlib import Path
import pymupdf as fitz
from reviewer import app, core

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = core.ROOT
        core.ROOT = Path(self.temp.name) / 'store'
        core.init()
        self.pdf = Path(self.temp.name) / 'synthetic.pdf'
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((60,60),'Treatment with interferon began in June 2003. Fatigue was reported.')
        doc.new_page()
        doc.save(self.pdf)
        doc.close()

    def tearDown(self):
        core.ROOT = self.previous
        self.temp.cleanup()

    def add_ocr(self, document_id, texts=None):
        texts = texts or {1:'Treatment with interferon began in June 2003. Fatigue was reported.',2:'OCR second page'}
        with core.connect() as db:
            pages=db.execute('SELECT id,number FROM pages WHERE document_id=? ORDER BY number',(document_id,)).fetchall()
            for page in pages:
                text=texts[page['number']]
                db.execute("INSERT INTO transcriptions(page_id,job_id,model,text,status,error,image_hash,created) VALUES(?,NULL,'mock',?,'complete',NULL,NULL,?)",(page['id'],text,core.now()))

    def test_ingestion_deduplicates_and_flags_scan(self):
        before = self.pdf.read_bytes()
        first = core.ingest(self.pdf)
        self.assertEqual(first,core.ingest(self.pdf))
        self.assertEqual(before,self.pdf.read_bytes())
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM pages').fetchone()[0],2)
            self.assertEqual([r[0] for r in db.execute('SELECT status FROM pages ORDER BY number')],['ready','needs_ocr'])
            self.assertEqual(db.execute("SELECT count(*) FROM page_search WHERE page_search MATCH 'interferon'").fetchone()[0],0)

    def test_quote_mismatch_is_flagged_without_discarding_page_candidates(self):
        did=core.ingest(self.pdf)
        self.add_ocr(did)
        with core.connect() as db:
            jid=db.execute('INSERT INTO jobs(document_id,status,model,created) VALUES(?,?,?,?)',(did,'queued','mock',core.now())).lastrowid
        events=[
            {'event_type':'treatment','treatment':'interferon','date_as_written':'June 2003','date_basis':'explicit','source_quote':'invented quote'},
            {'event_type':'symptom','treatment':'Fatigue','date_as_written':None,'date_basis':'absent','source_quote':'Fatigue was reported.'},
        ]
        def output(model,prompt):
            return json.dumps({'events':events}) if 'OCR second page' not in prompt else json.dumps({'events':[]})
        core.extract(jid,output)
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM evidence').fetchone()[0],2)
            self.assertEqual(db.execute('SELECT status FROM jobs WHERE id=?',(jid,)).fetchone()[0],'complete')
            self.assertEqual([r[0] for r in db.execute('SELECT quote_match FROM evidence ORDER BY id')],[0,1])
            self.assertIsNone(db.execute('SELECT start FROM evidence WHERE quote_match=0').fetchone()[0])
            self.assertIn('1 quotation mismatches flagged',db.execute('SELECT error FROM jobs WHERE id=?',(jid,)).fetchone()[0])
        core.extract(jid,lambda m,p:self.fail('Completed pages must not be run again'))
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM evidence').fetchone()[0],2)

    def test_large_html_table_is_rendered_chunked_and_quote_checked(self):
        table='<p>Laboratory report</p><table><tr><th>Test</th><th>Result</th></tr>'
        table+=''.join('<tr><td>Haemoglobin</td><td>120 g/L</td></tr>' for _ in range(400))
        table+='</table><p>End of report</p>'
        readable,was_rendered=core.readable_extraction_source(table)
        self.assertTrue(was_rendered)
        self.assertNotIn('<td>',readable)
        self.assertIn('Test: Haemoglobin | Result: 120 g/L',readable)
        chunks=core.split_extraction_source(readable)
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(len(chunk)<=core.EXTRACTION_CHUNK_CHARS for chunk in chunks))

        did=core.ingest(self.pdf)
        self.add_ocr(did,{1:table,2:'OCR second page'})
        with core.connect() as db:
            jid=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','mock-extractor',core.now(),'evidence')).lastrowid
        prompts=[]
        event={'event_type':'laboratory result','treatment':'Haemoglobin 120 g/L','date_as_written':None,'date_basis':'absent','source_quote':'Haemoglobin'}
        def call(model,prompt):
            prompts.append(prompt)
            return json.dumps({'events':[event] if 'Haemoglobin' in prompt else []})
        core.extract(jid,call)
        table_prompts=[prompt for prompt in prompts if 'Test: Haemoglobin' in prompt]
        self.assertEqual(len(table_prompts),len(chunks))
        self.assertTrue(all('<td>' not in prompt for prompt in table_prompts))
        with core.connect() as db:
            evidence=db.execute('SELECT quote,quote_match,quote_basis,start FROM evidence').fetchone()
            run=db.execute("SELECT status,output FROM runs WHERE job_id=? AND page_id=(SELECT id FROM pages WHERE document_id=? AND number=1)",(jid,did)).fetchone()
        self.assertEqual(tuple(evidence),('Haemoglobin',1,'readable_table',None))
        self.assertEqual(run['status'],'complete')
        self.assertEqual(len(json.loads(run['output'])['events']),1)

    def test_init_completes_old_fully_processed_jobs_with_quote_warnings(self):
        did=core.ingest(self.pdf)
        with core.connect() as db:
            pages=db.execute('SELECT id FROM pages WHERE document_id=? ORDER BY number',(did,)).fetchall()
            jid=db.execute("INSERT INTO jobs(document_id,status,model,created,kind,error) VALUES(?,?,?,?,?,?)",
                (did,'needs_attention','medgemma-test',core.now(),'medgemma','0 failed pages; 1 quotation mismatches flagged for review; 2 pages eligible')).lastrowid
            run_ids=[]
            for page in pages:
                run_ids.append(db.execute("INSERT INTO runs(job_id,page_id,model,output,status,created) VALUES(?,?,?,'{}','complete',?)",
                    (jid,page['id'],'medgemma-test',core.now())).lastrowid)
            db.execute("INSERT INTO evidence(run_id,page_id,payload,quote,quote_match) VALUES(?,?, '{}','mismatched quote',0)",(run_ids[0],pages[0]['id']))
        core.init()
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT status FROM jobs WHERE id=?',(jid,)).fetchone()[0],'complete')
            self.assertEqual(db.execute('SELECT quote_match FROM evidence').fetchone()[0],0)

    def test_medgemma_pass_adds_separate_candidates_without_deduplication(self):
        did=core.ingest(self.pdf)
        self.add_ocr(did)
        event={'event_type':'symptom','treatment':'Fatigue','date_as_written':None,'date_basis':'absent','source_quote':'Fatigue was reported.'}
        seen=[]
        def mock_call(model,prompt):
            seen.append((model,prompt))
            events=[] if 'OCR second page' in prompt else [event]
            return json.dumps({'events':events})
        with core.connect() as db:
            primary=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','nuextract-local',core.now(),'evidence')).lastrowid
            medgemma=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','medgemma-local',core.now(),'medgemma')).lastrowid
        core.extract(primary,mock_call)
        core.extract(medgemma,mock_call)
        with core.connect() as db:
            rows=db.execute('SELECT r.model,e.quote FROM evidence e JOIN runs r ON r.id=e.run_id ORDER BY e.id').fetchall()
            self.assertEqual([r['model'] for r in rows],['nuextract-local','medgemma-local'])
            self.assertEqual([r['quote'] for r in rows],['Fatigue was reported.']*2)
            self.assertIn(core.MEDGEMMA_PROMPT,seen[2][1])
            self.assertEqual(db.execute('SELECT output_token_limit FROM jobs WHERE id=?',(medgemma,)).fetchone()[0],2048)

    def test_extraction_requires_ocr_and_ignores_embedded_text(self):
        did=core.ingest(self.pdf)
        with core.connect() as db:
            jid=db.execute('INSERT INTO jobs(document_id,status,model,created) VALUES(?,?,?,?)',(did,'queued','mock',core.now())).lastrowid
        core.extract(jid,lambda m,p:self.fail('Must not extract from embedded PDF text'))
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM evidence').fetchone()[0],0)
            self.assertEqual(db.execute("SELECT count(*) FROM page_search WHERE page_search MATCH 'interferon'").fetchone()[0],0)

    def test_global_lexical_search_uses_latest_ocr_and_supports_scopes(self):
        did=core.ingest(self.pdf)
        self.add_ocr(did)
        core.rebuild_page_search()
        results=app.search('June fatigue',limit=100)
        self.assertEqual(len(results),1)
        self.assertEqual((results[0]['document_id'],results[0]['number']),(did,1))
        self.assertEqual(results[0]['document_name'],'synthetic.pdf')
        self.assertEqual(app.search('fatigue missingterm',limit=100),[])
        self.assertEqual(app.search('June fatigue',document_id='other-document',limit=100),[])

    def test_ignored_pages_are_excluded_from_ocr_extraction_and_lexical_search(self):
        did=core.ingest(self.pdf)
        with core.connect() as db:
            ignored=db.execute('SELECT id FROM pages WHERE document_id=? AND number=2',(did,)).fetchone()['id']
            db.execute('UPDATE pages SET ignored=1 WHERE id=?',(ignored,))
            ocr=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','ocr-model',core.now(),'ocr')).lastrowid
        calls=[]
        core.ocr(ocr,lambda model,path:calls.append(path.name) or 'kept page OCR text')
        self.assertEqual(calls,['1.png'])
        with core.connect() as db:
            primary=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','extractor',core.now(),'evidence')).lastrowid
        core.extract(primary,lambda model,prompt:'{"events":[]}')
        med_calls=[]
        with core.connect() as db:
            medgemma=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','medgemma-local',core.now(),'medgemma')).lastrowid
        core.extract(medgemma,lambda model,prompt:med_calls.append(prompt) or '{"events":[]}')
        self.assertEqual(len(med_calls),1)
        self.assertEqual(core.lexical_search('kept OCR',limit=100)[0]['number'],1)
        with core.connect() as db:
            self.assertEqual(db.execute("SELECT status,total_pages FROM jobs WHERE id=?",(ocr,)).fetchone()['status'],'complete')
            self.assertEqual(db.execute("SELECT count(*) FROM runs WHERE job_id=?",(primary,)).fetchone()[0],1)
            med_runs=db.execute("SELECT count(*) FROM runs WHERE job_id=?",(medgemma,)).fetchone()[0]
            self.assertEqual(med_runs,1)
            self.assertIsNone(db.execute("SELECT id FROM runs WHERE job_id=? AND page_id=?",(medgemma,ignored)).fetchone())
            db.execute("INSERT INTO transcriptions(page_id,model,text,status,created) VALUES(?,'manual-edit','private unique phrase','complete',?)",(ignored,core.now()))
            core.rebuild_page_search(db)
        self.assertEqual(core.lexical_search('unique phrase',limit=100),[])

    def test_ignored_page_with_existing_ocr_and_nuextract_is_skipped_by_later_stages(self):
        did=core.ingest(self.pdf)
        self.add_ocr(did)
        with core.connect() as db:
            ignored=db.execute('SELECT id FROM pages WHERE document_id=? AND number=2',(did,)).fetchone()['id']
            # Simulate a page that already completed NuExtract before being ignored.
            prior_job=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'complete','extractor',core.now(),'evidence')).lastrowid
            prior_run=db.execute("INSERT INTO runs(job_id,page_id,model,prompt,output,status,created) VALUES(?,?,?,?,?,?,?)",(prior_job,ignored,'extractor','prior prompt','{}','complete',core.now())).lastrowid
            db.execute("INSERT INTO evidence(run_id,page_id,payload,quote,quote_match) VALUES(?,?,'{}','OCR second page',1)",(prior_run,ignored))
            db.execute('UPDATE pages SET ignored=1 WHERE id=?',(ignored,))
            nuextract=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','extractor',core.now(),'evidence')).lastrowid
            medgemma=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','medgemma-local',core.now(),'medgemma')).lastrowid

        nuextract_calls=[]
        core.extract(nuextract,lambda model,prompt:nuextract_calls.append(prompt) or '{"events":[]}')
        medgemma_calls=[]
        core.extract(medgemma,lambda model,prompt:medgemma_calls.append(prompt) or '{"events":[]}')

        self.assertEqual(len(nuextract_calls),1)
        self.assertEqual(len(medgemma_calls),1)
        self.assertNotIn('OCR second page',nuextract_calls[0])
        self.assertNotIn('OCR second page',medgemma_calls[0])
        with core.connect() as db:
            self.assertIsNotNone(db.execute('SELECT id FROM runs WHERE id=?',(prior_run,)).fetchone())
            self.assertIsNone(db.execute('SELECT id FROM runs WHERE job_id=? AND page_id=?',(nuextract,ignored)).fetchone())
            self.assertIsNone(db.execute('SELECT id FROM runs WHERE job_id=? AND page_id=?',(medgemma,ignored)).fetchone())

    def test_manual_ocr_edit_is_latest_version_and_updates_search(self):
        did=core.ingest(self.pdf)
        self.add_ocr(did)
        with core.connect() as db:
            page=db.execute('SELECT id FROM pages WHERE document_id=? AND number=1',(did,)).fetchone()['id']
            before=db.execute('SELECT count(*) FROM transcriptions WHERE page_id=?',(page,)).fetchone()[0]
            db.execute("INSERT INTO transcriptions(page_id,job_id,model,text,status,created) VALUES(?,NULL,'manual-edit','corrected bespoke phrase','complete',?)",(page,core.now()))
            core.rebuild_page_search(db)
            self.assertEqual(db.execute('SELECT count(*) FROM transcriptions WHERE page_id=?',(page,)).fetchone()[0],before+1)
            self.assertEqual(db.execute('SELECT text FROM transcriptions WHERE page_id=? AND status=\'complete\' ORDER BY id DESC LIMIT 1',(page,)).fetchone()['text'],'corrected bespoke phrase')
        self.assertEqual(len(core.lexical_search('corrected bespoke',limit=100)),1)
        self.assertEqual(core.lexical_search('interferon',limit=100),[])

    def test_ignored_pages_are_excluded_from_ocr_extraction_and_lexical_search(self):
        did=core.ingest(self.pdf)
        with core.connect() as db:
            ignored=db.execute('SELECT id FROM pages WHERE document_id=? AND number=2',(did,)).fetchone()['id']
            db.execute('UPDATE pages SET ignored=1 WHERE id=?',(ignored,))
            ocr=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','ocr-model',core.now(),'ocr')).lastrowid
        calls=[]
        core.ocr(ocr,lambda model,path:calls.append(path.name) or 'kept page OCR text')
        self.assertEqual(calls,['1.png'])
        with core.connect() as db:
            primary=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','extractor',core.now(),'evidence')).lastrowid
        core.extract(primary,lambda model,prompt:'{"events":[]}')
        self.assertEqual(core.lexical_search('kept OCR',limit=100)[0]['number'],1)
        with core.connect() as db:
            self.assertEqual(db.execute("SELECT status,total_pages FROM jobs WHERE id=?",(ocr,)).fetchone()['status'],'complete')
            self.assertEqual(db.execute("SELECT count(*) FROM runs WHERE job_id=?",(primary,)).fetchone()[0],1)
            db.execute("INSERT INTO transcriptions(page_id,model,text,status,created) VALUES(?,'manual-edit','private unique phrase','complete',?)",(ignored,core.now()))
            core.rebuild_page_search(db)
        self.assertEqual(core.lexical_search('unique phrase',limit=100),[])

    def test_manual_ocr_edit_is_latest_version_and_updates_search(self):
        did=core.ingest(self.pdf)
        self.add_ocr(did)
        with core.connect() as db:
            page=db.execute('SELECT id FROM pages WHERE document_id=? AND number=1',(did,)).fetchone()['id']
            before=db.execute('SELECT count(*) FROM transcriptions WHERE page_id=?',(page,)).fetchone()[0]
            db.execute("INSERT INTO transcriptions(page_id,job_id,model,text,status,created) VALUES(?,NULL,'manual-edit','corrected bespoke phrase','complete',?)",(page,core.now()))
            core.rebuild_page_search(db)
            self.assertEqual(db.execute('SELECT count(*) FROM transcriptions WHERE page_id=?',(page,)).fetchone()[0],before+1)
            self.assertEqual(db.execute('SELECT text FROM transcriptions WHERE page_id=? AND status=\'complete\' ORDER BY id DESC LIMIT 1',(page,)).fetchone()['text'],'corrected bespoke phrase')
        self.assertEqual(len(core.lexical_search('corrected bespoke',limit=100)),1)
        self.assertEqual(core.lexical_search('interferon',limit=100),[])

    def test_extraction_persists_lmstudio_token_usage(self):
        did=core.ingest(self.pdf)
        self.add_ocr(did)
        with core.connect() as db:
            jid=db.execute('INSERT INTO jobs(document_id,status,model,created,provider) VALUES(?,?,?,?,?)',(did,'queued','mock',core.now(),'lmstudio')).lastrowid
        def model_result(model,prompt):
            return {'content':'{"events":[]}','input_tokens':1876,'output_tokens':9,'finish_reason':'stop'}
        core.extract(jid,model_result)
        with core.connect() as db:
            run=db.execute('SELECT input_tokens,output_tokens,finish_reason FROM runs WHERE job_id=? AND status=\'complete\' ORDER BY id LIMIT 1',(jid,)).fetchone()
        self.assertEqual(tuple(run),(1876,9,'stop'))

    def test_output_limit_stops_extraction_before_later_pages(self):
        three_page_pdf=Path(self.temp.name)/'three-pages.pdf'
        document=fitz.open(self.pdf)
        page=document.new_page()
        page.insert_text((60,60),'Third page OCR')
        document.save(three_page_pdf)
        document.close()
        did=core.ingest(three_page_pdf)
        self.add_ocr(did,{1:'First page OCR',2:'Second page OCR',3:'Third page OCR'})
        with core.connect() as db:
            jid=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','medgemma-test',core.now(),'medgemma')).lastrowid
        calls=[]
        def call(model,prompt):
            calls.append(prompt)
            if 'Second page OCR' in prompt:
                raise core.OutputLimitError('finish_reason=length',content='{"events":[',input_tokens=468,output_tokens=12000)
            return json.dumps({'events':[]})
        core.extract(jid,call)
        self.assertEqual(len(calls),2)
        with core.connect() as db:
            rows=db.execute('SELECT p.number,r.status FROM pages p LEFT JOIN runs r ON r.page_id=p.id AND r.job_id=? WHERE p.document_id=? ORDER BY p.number',(jid,did)).fetchall()
            self.assertEqual([tuple(row) for row in rows],[(1,'complete'),(2,'failed'),(3,None)])
            failed_run=db.execute("SELECT output,input_tokens,output_tokens,finish_reason FROM runs WHERE job_id=? AND status='failed'",(jid,)).fetchone()
            self.assertEqual(tuple(failed_run),('{"events":[',468,12000,'length'))
            self.assertEqual(db.execute('SELECT output FROM failed_attempts WHERE job_id=?',(jid,)).fetchone()['output'],'{"events":[')
            self.assertEqual(tuple(db.execute('SELECT max_input_tokens,max_output_tokens FROM jobs WHERE id=?',(jid,)).fetchone()),(468,12000))
            job=db.execute('SELECT status,completed_pages,error FROM jobs WHERE id=?',(jid,)).fetchone()
        self.assertEqual((job['status'],job['completed_pages']),('needs_attention',1))
        self.assertIn('Stopped after page 2',job['error'])
        self.assertIn('later pages were not processed',job['error'])
        self.assertEqual(next(run for run in app.runs(did) if run['status']=='failed')['output'],'{"events":[')
        core.clear_stage_data(did,'medgemma')
        with core.connect() as db:
            jid=db.execute("INSERT INTO jobs(document_id,status,model,created,kind,stop_on_output_limit) VALUES(?,?,?,?,?,0)",(did,'queued','medgemma-test',core.now(),'medgemma')).lastrowid
        calls=[]
        def continue_call(model,prompt):
            calls.append(prompt)
            if 'Second page OCR' in prompt:
                raise core.OutputLimitError('finish_reason=length')
            return json.dumps({'events':[]})
        core.extract(jid,continue_call)
        self.assertEqual(len(calls),3)
        with core.connect() as db:
            rows=db.execute('SELECT p.number,r.status FROM pages p LEFT JOIN runs r ON r.page_id=p.id AND r.job_id=? WHERE p.document_id=? ORDER BY p.number',(jid,did)).fetchall()
            self.assertEqual([tuple(row) for row in rows],[(1,'complete'),(2,'failed'),(3,'complete')])
            job=db.execute('SELECT status,error FROM jobs WHERE id=?',(jid,)).fetchone()
        self.assertEqual(job['status'],'needs_attention')
        self.assertIn('Continued after finish_reason=length on page(s) 2',job['error'])

    def test_malformed_output_rejected(self):
        for output in ['not json','{}','{"events":[{}]}']:
            with self.assertRaises((ValueError,TypeError)):
                core.validate(output,'source')

    def test_existing_database_migrates_quote_match_flag(self):
        legacy_root=Path(self.temp.name)/'legacy'
        legacy_root.mkdir()
        legacy_db=legacy_root/'records.sqlite'
        import sqlite3
        with sqlite3.connect(legacy_db) as db:
            db.execute('CREATE TABLE documents(id TEXT PRIMARY KEY,name TEXT,path TEXT,pages INTEGER,status TEXT,error TEXT)')
            db.execute("CREATE TABLE evidence(id INTEGER PRIMARY KEY,run_id INTEGER,page_id TEXT,payload TEXT,quote TEXT,start INTEGER,end INTEGER,status TEXT DEFAULT 'unreviewed',notes TEXT DEFAULT '')")
            db.execute('CREATE TABLE batches(id INTEGER PRIMARY KEY,folder TEXT,status TEXT,ocr_model TEXT,evidence_model TEXT,created TEXT,updated TEXT,total INTEGER,completed INTEGER DEFAULT 0)')
            db.execute('CREATE TABLE batch_items(id INTEGER PRIMARY KEY,batch_id INTEGER,path TEXT,name TEXT,status TEXT,document_id TEXT,ocr_job_id INTEGER,evidence_job_id INTEGER,error TEXT,pages INTEGER DEFAULT 0,transcribed INTEGER DEFAULT 0,candidates INTEGER DEFAULT 0)')
            db.execute("INSERT INTO batches(id,folder,status,created,updated,total) VALUES(1,'legacy','complete','2026-01-01T10:00:00+00:00','2026-01-01T10:05:00+00:00',0)")
        core.ROOT=legacy_root
        core.init()
        with core.connect() as db:
            document_columns={row['name'] for row in db.execute('PRAGMA table_info(documents)')}
            evidence_columns={row['name'] for row in db.execute('PRAGMA table_info(evidence)')}
            batch_columns={row['name'] for row in db.execute('PRAGMA table_info(batches)')}
            item_columns={row['name'] for row in db.execute('PRAGMA table_info(batch_items)')}
            self.assertIn('source_ext',document_columns)
            self.assertIn('quote_match',evidence_columns)
            self.assertIn('quote_basis',evidence_columns)
            self.assertIn('medgemma_model',batch_columns)
            self.assertIn('started_at',batch_columns)
            self.assertIn('finished_at',batch_columns)
            legacy_batch=db.execute('SELECT started_at,finished_at FROM batches WHERE id=1').fetchone()
            self.assertEqual((legacy_batch['started_at'],legacy_batch['finished_at']),('2026-01-01T10:00:00+00:00','2026-01-01T10:05:00+00:00'))
            self.assertIn('medgemma_job_id',item_columns)

    def test_import_only_then_manual_pages(self):
        did=core.ingest(self.pdf,register_only=True)
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM pages').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT status FROM documents').fetchone()[0],'imported')
        core.ingest(core.ROOT/'originals'/(did+'.pdf'))
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM pages').fetchone()[0],2)
            self.assertEqual(db.execute('SELECT path FROM documents').fetchone()[0],str(self.pdf))

    def test_delete_all_related_data_and_reimport(self):
        did=core.ingest(self.pdf)
        self.add_ocr(did)
        with core.connect() as db:
            jid=db.execute('INSERT INTO jobs(document_id,status,model,created) VALUES(?,?,?,?)',(did,'queued','mock',core.now())).lastrowid
        core.extract(jid,lambda m,p:'bad json')
        event={'event_type':'symptom','treatment':'Fatigue','date_as_written':None,'date_basis':'absent','source_quote':'Fatigue was reported.'}
        core.extract(jid,lambda m,p:json.dumps({'events':[event]}))
        with core.connect() as db:
            eid=db.execute('SELECT id FROM evidence').fetchone()[0]
            db.execute('INSERT INTO decisions(evidence_id,status,notes,created) VALUES(?,?,?,?)',(eid,'confirmed','test',core.now()))
        # Another document must survive the targeted deletion.
        other=self.pdf.with_name('other.pdf')
        doc=fitz.open();doc.new_page();doc.save(other);doc.close()
        other_id=core.ingest(other)
        core.delete_document(did)
        self.assertTrue(self.pdf.exists())
        self.assertFalse((core.ROOT/'originals'/(did+'.pdf')).exists())
        self.assertFalse((core.ROOT/'pages'/did).exists())
        with core.connect() as db:
            for table in ('jobs','runs','evidence','decisions','failed_attempts'):
                self.assertEqual(db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
            self.assertEqual(db.execute('SELECT id FROM documents').fetchone()[0],other_id)
            self.assertEqual(db.execute("SELECT count(*) FROM page_search WHERE page_search MATCH 'interferon'").fetchone()[0],0)
            self.assertEqual(list(db.execute('PRAGMA foreign_key_check')),[])
        self.assertEqual(core.ingest(self.pdf),did)

    def test_delete_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            core.delete_document('../outside')

    def test_ocr_versions_retry_selection_and_deletion(self):
        did=core.ingest(self.pdf)
        with core.connect() as db:
            jid=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','lighton-test',core.now(),'ocr')).lastrowid
        def partial(model,path):
            if path.name=='2.png':
                raise ValueError('Simulated failure')
            return 'Fresh OCR text'
        core.ocr(jid,partial)
        calls=[]
        core.ocr(jid,lambda m,p: calls.append(p.name) or 'Second page transcription')
        self.assertEqual(calls,['2.png'])
        with core.connect() as db:
            rows=db.execute("SELECT * FROM transcriptions WHERE status='complete' ORDER BY id").fetchall()
            self.assertEqual(len(rows),2)
            self.assertIn('interferon',db.execute('SELECT text FROM pages WHERE number=1').fetchone()[0])
            self.assertEqual(db.execute("SELECT count(*) FROM page_search WHERE page_search MATCH 'Fresh'").fetchone()[0],1)
            self.assertEqual(db.execute("SELECT count(*) FROM page_search WHERE page_search MATCH 'Second'").fetchone()[0],1)
            self.assertEqual(db.execute("SELECT count(*) FROM page_search WHERE page_search MATCH 'interferon'").fetchone()[0],0)
        core.delete_document(did)
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM transcriptions').fetchone()[0],0)

    def test_empty_ocr_is_flagged_as_suspected_blank_page_for_review(self):
        did=core.ingest(self.pdf)
        with core.connect() as db:
            jid=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','lighton-test',core.now(),'ocr')).lastrowid
        core.ocr(jid,lambda model,path:'' if path.name=='1.png' else 'recognized content')
        with core.connect() as db:
            job=db.execute('SELECT status,error FROM jobs WHERE id=?',(jid,)).fetchone()
            first=db.execute("SELECT status,error FROM transcriptions WHERE page_id=(SELECT id FROM pages WHERE document_id=? AND number=1) ORDER BY id DESC LIMIT 1",(did,)).fetchone()
        self.assertEqual(job['status'],'needs_attention')
        self.assertIn('Suspected Blank Page',job['error'])
        self.assertIn('Suspected Blank Page',first['error'])
        self.assertIn('mark it Ignore page',first['error'])

    def test_stage_progress_tracks_completed_pages_and_resume(self):
        did=core.ingest(self.pdf)
        with core.connect() as db:
            jid=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','lighton-test',core.now(),'ocr')).lastrowid
        core.ocr(jid,lambda m,p:'first OCR' if p.name=='1.png' else (_ for _ in ()).throw(ValueError('retry me')))
        with core.connect() as db:
            job=db.execute('SELECT completed_pages,total_pages,last_completed_page,current_page,runtime_seconds FROM jobs WHERE id=?',(jid,)).fetchone()
            self.assertEqual(tuple(job[:4]),(1,2,1,None))
            first_runtime=job['runtime_seconds']
        calls=[]
        core.ocr(jid,lambda m,p:calls.append(p.name) or 'second OCR')
        self.assertEqual(calls,['2.png'])
        with core.connect() as db:
            job=db.execute('SELECT completed_pages,total_pages,last_completed_page,status,started_at,finished_at,error,runtime_seconds,runtime_started_at FROM jobs WHERE id=?',(jid,)).fetchone()
            self.assertEqual(tuple(job[:4]),(2,2,2,'complete'))
            self.assertIsNotNone(job['started_at'])
            self.assertIsNotNone(job['finished_at'])
            self.assertGreaterEqual(job['finished_at'],job['started_at'])
            self.assertGreater(job['runtime_seconds'],first_runtime)
            self.assertIsNone(job['runtime_started_at'])
            self.assertIsNone(job['error'])

    def test_init_backfills_progress_for_existing_jobs_from_saved_pages(self):
        did=core.ingest(self.pdf)
        with core.connect() as db:
            page=db.execute('SELECT id FROM pages WHERE document_id=? AND number=1',(did,)).fetchone()['id']
            ocr=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'interrupted','ocr-model',core.now(),'ocr')).lastrowid
            db.execute("INSERT INTO transcriptions(page_id,job_id,model,text,status,created) VALUES(?,?,?,'saved OCR','complete',?)",(page,ocr,'ocr-model',core.now()))
            evidence=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'interrupted','extractor',core.now(),'evidence')).lastrowid
            db.execute("INSERT INTO runs(job_id,page_id,model,prompt,output,status,created) VALUES(?,?,?,'prompt','{}','complete',?)",(evidence,page,'extractor',core.now()))
        core.init()
        with core.connect() as db:
            ocr_progress=db.execute('SELECT completed_pages,total_pages,last_completed_page FROM jobs WHERE id=?',(ocr,)).fetchone()
            evidence_progress=db.execute('SELECT completed_pages,total_pages,last_completed_page FROM jobs WHERE id=?',(evidence,)).fetchone()
            migrated_timing=db.execute('SELECT started_at,finished_at FROM jobs WHERE id=?',(ocr,)).fetchone()
        self.assertEqual(tuple(ocr_progress),(1,2,1))
        self.assertEqual(tuple(evidence_progress),(1,2,1))
        self.assertIsNotNone(migrated_timing['started_at'])
        self.assertIsNotNone(migrated_timing['finished_at'])

    def test_clear_stage_data_preserves_independent_stages_and_cascades_ocr(self):
        did=core.ingest(self.pdf)
        self.add_ocr(did)
        event={'event_type':'symptom','treatment':'Fatigue','date_as_written':None,'date_basis':'absent','source_quote':'Fatigue was reported.'}
        with core.connect() as db:
            primary=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','nuextract-local',core.now(),'evidence')).lastrowid
            med=db.execute("INSERT INTO jobs(document_id,status,model,created,kind) VALUES(?,?,?,?,?)",(did,'queued','medgemma-local',core.now(),'medgemma')).lastrowid
        core.extract(primary,lambda m,p:json.dumps({'events':[event]}))
        core.extract(med,lambda m,p:json.dumps({'events':[event]}))
        with core.connect() as db:
            eid=db.execute("SELECT e.id FROM evidence e JOIN runs r ON r.id=e.run_id WHERE r.job_id=? LIMIT 1",(primary,)).fetchone()[0]
            db.execute('INSERT INTO decisions(evidence_id,status,notes,created) VALUES(?,?,?,?)',(eid,'confirmed','keep',core.now()))
        core.clear_stage_data(did,'nuextract')
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM transcriptions').fetchone()[0],2)
            self.assertEqual(db.execute('SELECT count(*) FROM evidence e JOIN runs r ON r.id=e.run_id WHERE r.job_id=?',(primary,)).fetchone()[0],0)
            self.assertEqual(db.execute('SELECT count(*) FROM evidence e JOIN runs r ON r.id=e.run_id WHERE r.job_id=?',(med,)).fetchone()[0],2)
        core.clear_stage_data(did,'ocr')
        with core.connect() as db:
            for table in ('jobs','runs','evidence','decisions','transcriptions','page_embeddings','semantic_page_index'):
                self.assertEqual(db.execute('SELECT count(*) FROM '+table).fetchone()[0],0,table)
            self.assertEqual(db.execute('SELECT count(*) FROM pages').fetchone()[0],2)

if __name__=='__main__':
    unittest.main()
