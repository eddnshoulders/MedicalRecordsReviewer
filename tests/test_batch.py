import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf as fitz
from reviewer import app, core


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.previous=core.ROOT
        core.ROOT=Path(self.temp.name)/'store'
        core.init()

    def tearDown(self):
        core.ROOT=self.previous
        self.temp.cleanup()

    def test_medgemma_text_cannot_be_selected_as_ocr_model(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as error:
            app.validate_ocr_model('medgemma-27b-text-it')
        self.assertIn('not an OCR model',error.exception.detail)

    def test_completed_stage_data_blocks_new_stage_run(self):
        from fastapi import HTTPException
        source=Path(self.temp.name)/'source.pdf'
        doc=fitz.open()
        doc.new_page()
        doc.save(source)
        doc.close()
        document_id=core.ingest(source)
        with core.connect() as db:
            page=db.execute('SELECT id FROM pages WHERE document_id=?',(document_id,)).fetchone()['id']
            ocr_job=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'complete','ocr','now','lmstudio','ocr')",(document_id,)).lastrowid
            db.execute("INSERT INTO transcriptions(page_id,job_id,model,text,status,created) VALUES(?,?,'ocr','text','complete','now')",(page,ocr_job))
            with self.assertRaises(HTTPException) as error:
                app.ensure_stage_empty(db,document_id,'ocr')
            self.assertIn('Delete OCR data',error.exception.detail)
            app.ensure_stage_empty(db,document_id,'evidence')

    def test_stop_endpoint_sets_cooperative_stop_flag(self):
        from fastapi import HTTPException
        source=Path(self.temp.name)/'stop.pdf'
        doc=fitz.open();doc.new_page();doc.save(source);doc.close()
        document_id=core.ingest(source)
        with core.connect() as db:
            db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'running','mock',?,'lmstudio','ocr')",(document_id,core.now()))
        app.stop_event.clear()
        try:
            response=app.stop_current_task()
            self.assertTrue(response['stop_requested'])
            self.assertTrue(app.stop_event.is_set())
        finally:
            app.stop_event.clear()

    def test_import_endpoint_prepares_pages_even_when_ocr_is_not_requested(self):
        source=Path(self.temp.name)/'imported.pdf'
        doc=fitz.open();doc.new_page();doc.save(source);doc.close()
        result=app.import_pdf(app.Import(path=str(source),register_only=True))
        with core.connect() as db:
            rows=db.execute('SELECT status,image FROM pages WHERE document_id=?',(result['id'],)).fetchall()
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['status'],'needs_ocr')
        self.assertTrue(Path(rows[0]['image']).is_file())

    def test_import_root_path_is_persisted_in_sqlite(self):
        selected=Path(self.temp.name).resolve()
        result=app.save_import_root(app.ImportRoot(path=str(selected)))
        self.assertEqual(result,{'path':str(selected)})
        self.assertEqual(app.import_root_setting(),{'path':str(selected)})

    def test_missing_page_preview_is_rebuilt_without_clearing_processing_data(self):
        source=Path(self.temp.name)/'preview.pdf'
        doc=fitz.open();page=doc.new_page();page.insert_text((72,72),'Synthetic preview page');doc.save(source);doc.close()
        document_id=core.ingest(source)
        with core.connect() as db:
            page=db.execute('SELECT id,image FROM pages WHERE document_id=?',(document_id,)).fetchone()
            ocr_id=db.execute("INSERT INTO transcriptions(page_id,model,text,status,created) VALUES(?,'mock-ocr','preserved text','complete',?)",(page['id'],core.now())).lastrowid
            job_id=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'complete','mock-extractor',?,'lmstudio','evidence')",(document_id,core.now())).lastrowid
            run_id=db.execute("INSERT INTO runs(job_id,page_id,model,output,status,created,transcription_id) VALUES(?,?,?,'{}','complete',?,?)",(job_id,page['id'],'mock-extractor',core.now(),ocr_id)).lastrowid
            db.execute("INSERT INTO evidence(run_id,page_id,payload,quote,status) VALUES(?,?,'{}','preserved quote','confirmed')",(run_id,page['id']))
        Path(page['image']).unlink()
        preview=app.image(page['id'])
        rebuilt=Path(preview.path)
        self.assertTrue(rebuilt.is_file())
        self.assertEqual(rebuilt.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT text FROM transcriptions WHERE id=?',(ocr_id,)).fetchone()['text'],'preserved text')
            self.assertEqual(db.execute('SELECT count(*) FROM evidence WHERE run_id=?',(run_id,)).fetchone()[0],1)
            self.assertEqual(Path(db.execute('SELECT image FROM pages WHERE id=?',(page['id'],)).fetchone()['image']),rebuilt)

    def test_main_stage_button_resumes_incomplete_job(self):
        from unittest.mock import patch
        source=Path(self.temp.name)/'resume.pdf'
        doc=fitz.open();doc.new_page();doc.new_page();doc.save(source);doc.close()
        document_id=core.ingest(source)
        with core.connect() as db:
            pages=db.execute('SELECT id FROM pages WHERE document_id=? ORDER BY number',(document_id,)).fetchall()
            job_id=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'needs_attention','ocr-model',?,'lmstudio','ocr')",(document_id,core.now())).lastrowid
            db.execute("INSERT INTO transcriptions(page_id,job_id,model,text,status,created) VALUES(?,?,'ocr-model','page one','complete',?)",(pages[0]['id'],job_id,core.now()))
        class BackgroundThread:
            def __init__(self,*args,**kwargs): pass
            def start(self): app.lock.release()
        with patch.object(app,'validate_loaded_model'), patch.object(app.threading,'Thread',BackgroundThread):
            result=app.workflow_selected(app.SelectedWorkflow(stage='ocr',document_ids=[document_id],model='ocr-model'))
        self.assertEqual(result,{'job_ids':[job_id]})
        with core.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM jobs WHERE document_id=?',(document_id,)).fetchone()[0],1)
            self.assertEqual(db.execute('SELECT status FROM jobs WHERE id=?',(job_id,)).fetchone()['status'],'queued')

    def test_page_list_status_reflects_saved_ocr_not_import_status(self):
        source=Path(self.temp.name)/'source.pdf'
        doc=fitz.open()
        doc.new_page()
        doc.new_page()
        doc.save(source)
        doc.close()
        document_id=core.ingest(source)
        with core.connect() as db:
            pages=db.execute('SELECT id,status FROM pages WHERE document_id=? ORDER BY number',(document_id,)).fetchall()
            self.assertTrue(all(page['status']=='needs_ocr' for page in pages))
            db.execute("INSERT INTO transcriptions(page_id,model,text,status,created) VALUES(?,'mock-ocr','recognized text','complete',?)",(pages[0]['id'],core.now()))
            db.execute("INSERT INTO transcriptions(page_id,model,text,status,error,created) VALUES(?,'mock-ocr','','failed','synthetic OCR error',?)",(pages[1]['id'],core.now()))
            primary=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'complete','extractor',?,'lmstudio','evidence')",(document_id,core.now())).lastrowid
            med=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind,current_page) VALUES(?,'running','medgemma',?,'lmstudio','medgemma',1)",(document_id,core.now())).lastrowid
            db.execute("INSERT INTO runs(job_id,page_id,model,output,status,created) VALUES(?,?,?,'{}','complete',?)",(primary,pages[0]['id'],'extractor',core.now()))
            db.execute("INSERT INTO runs(job_id,page_id,model,output,status,error,created) VALUES(?,?,?,'','failed','synthetic extraction error',?)",(primary,pages[1]['id'],'extractor',core.now()))
        states=app.pages(document_id)
        self.assertEqual((states[0]['ocr_progress'],states[0]['nuextract_progress'],states[0]['medgemma_progress']),('complete','complete','running'))
        self.assertEqual((states[1]['ocr_progress'],states[1]['nuextract_progress'],states[1]['medgemma_progress']),('failed','failed','pending'))

    def test_page_ignore_and_manual_transcription_edit_api(self):
        from fastapi import HTTPException
        source=Path(self.temp.name)/'source.pdf'
        doc=fitz.open();doc.new_page();doc.save(source);doc.close()
        document_id=core.ingest(source)
        page_id=core.get_document_page_map(document_id)['pages'][0]['page_id']
        with core.connect() as db:
            transcription=db.execute("INSERT INTO transcriptions(page_id,model,text,status,created) VALUES(?,'mock-ocr','old OCR text','complete',?)",(page_id,core.now())).lastrowid
            job_id=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,'needs_attention','mock-extractor',?,'lmstudio','evidence')",(document_id,core.now())).lastrowid
            run_id=db.execute("INSERT INTO runs(job_id,page_id,model,output,status,error,created) VALUES(?,?,?,'','failed','synthetic oversized HTML table',?)",(job_id,page_id,'mock-extractor',core.now())).lastrowid
            evidence_id=db.execute("INSERT INTO evidence(run_id,page_id,payload,quote,quote_match,status,notes) VALUES(?,?,'{}','old OCR text',1,'confirmed','old review')",(run_id,page_id)).lastrowid
            db.execute("INSERT INTO decisions(evidence_id,status,notes,created) VALUES(?,'confirmed','old review',?)",(evidence_id,core.now()))
            db.execute("INSERT INTO failed_attempts(job_id,page_id,output,error,created) VALUES(?,?,?,'synthetic oversized HTML table',?)",(job_id,page_id,'partial output',core.now()))
            db.execute("INSERT INTO page_embeddings(page_id,transcription_id,model,chunk_index,chunk_text,vector,created) VALUES(?,?,'mock-embedding',0,'old OCR text','[]',?)",(page_id,transcription,core.now()))
            db.execute("INSERT INTO semantic_page_index(page_id,transcription_id,model,chunk_count,created) VALUES(?,?,'mock-embedding',1,?)",(page_id,transcription,core.now()))
            db.execute("INSERT INTO page_search(page_id,text) VALUES(?,'old OCR text')",(page_id,))
            batch_id=db.execute("INSERT INTO batches(folder,status,created,updated,total,completed) VALUES(?,'needs_attention',?,?,1,1)",(str(self.temp.name),core.now(),core.now())).lastrowid
            db.execute("INSERT INTO batch_items(batch_id,path,name,status,document_id,error,pages,transcribed,candidates) VALUES(?,?,?,'needs_attention',?,'Evidence page 1: synthetic oversized HTML table',1,1,1)",(batch_id,str(source),source.name,document_id))
        self.assertEqual(app.set_page_ignored(page_id,app.PageIgnore(ignored=True)),{'page_id':page_id,'ignored':True})
        states=app.pages(document_id)
        self.assertTrue(states[0]['ignored'])
        self.assertEqual(states[0]['ocr_progress'],'ignored')
        with core.connect() as db:
            for table in ('transcriptions','runs','evidence','decisions','failed_attempts','page_embeddings','semantic_page_index','page_search'):
                self.assertEqual(db.execute(f'SELECT count(*) FROM {table} WHERE '+('page_id=?' if table not in ('decisions',) else 'evidence_id=?'),(page_id if table!='decisions' else evidence_id,)).fetchone()[0],0,table)
            job=db.execute('SELECT status,total_pages,completed_pages,error FROM jobs WHERE id=?',(job_id,)).fetchone()
            self.assertEqual((job['status'],job['total_pages'],job['completed_pages']),('complete',0,0))
            self.assertIn('0 failed pages',job['error'])
            item=db.execute('SELECT status,error,transcribed,candidates FROM batch_items WHERE batch_id=?',(batch_id,)).fetchone()
            self.assertEqual((item['status'],item['error'],item['transcribed'],item['candidates']),('complete',None,0,0))
            self.assertEqual(db.execute('SELECT status FROM batches WHERE id=?',(batch_id,)).fetchone()['status'],'complete')
        self.assertEqual(app.runs(document_id),[])
        self.assertEqual(core.get_source(document_id,[1]),[])
        with self.assertRaises(HTTPException):
            app.edit_transcription(page_id,app.TranscriptionEdit(text='Should not save while ignored'))
        app.set_page_ignored(page_id,app.PageIgnore(ignored=False))
        self.assertTrue(app.edit_transcription(page_id,app.TranscriptionEdit(text='Corrected OCR text'))['saved'])
        with core.connect() as db:
            versions=db.execute('SELECT model,text FROM transcriptions WHERE page_id=? ORDER BY id',(page_id,)).fetchall()
        self.assertEqual((versions[0]['model'],versions[0]['text']),('manual-edit','Corrected OCR text'))
        self.assertEqual(core.get_source(document_id,[1])[0]['source_text'],'Corrected OCR text')
        with self.assertRaises(HTTPException):
            app.edit_transcription(page_id,app.TranscriptionEdit(text='  '))

    def test_batch_continues_after_bad_file_and_processes_image(self):
        folder=Path(self.temp.name)/'input'
        folder.mkdir()
        (folder/'a-broken.pdf').write_bytes(b'not a pdf')
        image_path=folder/'b-image.png'
        doc=fitz.open()
        doc.new_page(width=300,height=600).get_pixmap(dpi=96).save(image_path)
        doc.close()
        second_image_path=folder/'c-image.png'
        doc=fitz.open()
        doc.new_page(width=300,height=600).get_pixmap(dpi=96).save(second_image_path)
        doc.close()
        with core.connect() as db:
            batch_id=db.execute('INSERT INTO batches(folder,status,ocr_model,evidence_model,medgemma_model,created,updated,total) VALUES(?,?,?,?,?,?,?,?)',
                (str(folder),'queued','mock-ocr','mock-evidence','mock-medgemma',core.now(),core.now(),3)).lastrowid
            db.executemany('INSERT INTO batch_items(batch_id,path,name,status) VALUES(?,?,?,?)',
                [(batch_id,str(path),path.name,'queued') for path in sorted(folder.iterdir())])

        phases=[]
        def fake_ocr(job_id):
            with core.connect() as db:
                job=db.execute('SELECT document_id FROM jobs WHERE id=?',(job_id,)).fetchone()
                phases.append('ocr')
                pages=db.execute('SELECT id FROM pages WHERE document_id=?',(job['document_id'],)).fetchall()
                for page in pages:
                    db.execute("INSERT INTO transcriptions(page_id,job_id,model,text,status,error,image_hash,created) VALUES(?,?,?,'OCR text','complete',NULL,NULL,?)",
                        (page['id'],job_id,'mock-ocr',core.now()))
                db.execute("UPDATE jobs SET status='complete' WHERE id=?",(job_id,))

        def fake_extract(job_id):
            with core.connect() as db:
                job=db.execute('SELECT kind,document_id,model FROM jobs WHERE id=?',(job_id,)).fetchone()
                phases.append(job['kind'])
                for page in db.execute('SELECT id FROM pages WHERE document_id=?',(job['document_id'],)).fetchall():
                    db.execute("INSERT INTO runs(job_id,page_id,model,output,status,created) VALUES(?,?,?,'{}','complete',?)",
                        (job_id,page['id'],job['model'],core.now()))
                db.execute("UPDATE jobs SET status='complete' WHERE id=?",(job_id,))

        self.assertTrue(app.lock.acquire(False))
        with patch.object(core,'ocr',fake_ocr),patch.object(core,'extract',fake_extract):
            app.run_batch(batch_id)

        with core.connect() as db:
            batch=db.execute('SELECT status,completed,started_at,finished_at FROM batches WHERE id=?',(batch_id,)).fetchone()
            items=db.execute('SELECT name,status,error,document_id FROM batch_items WHERE batch_id=? ORDER BY name',(batch_id,)).fetchall()
            self.assertEqual((batch['status'],batch['completed']),('needs_attention',3))
            self.assertTrue(batch['started_at'])
            self.assertTrue(batch['finished_at'])
            self.assertGreaterEqual(batch['finished_at'],batch['started_at'])
            self.assertEqual(items[0]['status'],'failed')
            self.assertTrue(items[0]['error'])
            self.assertEqual(items[1]['status'],'complete')
            image_doc=db.execute('SELECT source_ext FROM documents WHERE id=?',(items[1]['document_id'],)).fetchone()
            self.assertEqual(image_doc['source_ext'],'.png')
            image_page=db.execute('SELECT image,status FROM pages WHERE document_id=?',(items[1]['document_id'],)).fetchone()
            self.assertEqual(image_page['status'],'needs_ocr')
            rendered=fitz.open(image_page['image'])
            self.assertEqual((rendered[0].get_pixmap(dpi=96).width,rendered[0].get_pixmap(dpi=96).height),(400,800))
            rendered.close()
            original=core.ROOT/'originals'/(items[1]['document_id']+'.png')
            self.assertTrue(original.is_file())
            self.assertEqual(items[2]['status'],'complete')
            self.assertEqual(phases,['ocr','evidence','medgemma'])
        file_messages=app.document_messages(items[1]['document_id'])
        self.assertEqual(len(file_messages),2)
        self.assertTrue(all(row['batch_id']==batch_id for row in file_messages))
        self.assertTrue(all(row['file_status']=='complete' for row in file_messages))
        core.delete_document(items[1]['document_id'])
        self.assertFalse(original.exists())


if __name__=='__main__':
    unittest.main()
