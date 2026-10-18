import json
import re
import threading
import urllib.error
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware
from . import core

lock = threading.Lock()
lm_startup_status={'available':False,'models':[],'error':'Not checked yet.'}

@asynccontextmanager
async def lifespan(app):
    core.init()
    try:
        lm_startup_status.update(available=True,models=core.list_models(),error=None)
    except Exception as exc:
        lm_startup_status.update(available=False,models=[],error=f'LM Studio was not reachable when the reviewer started: {exc}')
    with core.connect() as db:
        interrupted=[r['id'] for r in db.execute("SELECT id FROM jobs WHERE status IN ('queued','running')")]
        for job_id in interrupted:
            core.finish_job(db,job_id,'interrupted','Reviewer stopped before this stage finished.')
        db.execute("UPDATE batches SET status='interrupted',updated=?,finished_at=? WHERE status IN ('queued','running')",(core.now(),core.now()))
        db.execute("UPDATE semantic_jobs SET status='interrupted',updated=? WHERE status IN ('queued','running')",(core.now(),))
        db.execute("UPDATE batch_items SET status='failed',error=COALESCE(error,'Reviewer stopped before this file finished.') WHERE status IN ('queued','preparing','ocr','extracting')")
        db.execute("UPDATE batches SET completed=(SELECT count(*) FROM batch_items i WHERE i.batch_id=batches.id AND i.status IN ('complete','needs_attention','failed')) WHERE status='interrupted'")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1','localhost','testserver'])

@app.middleware('http')
async def local_only(request: Request, call_next):
    if request.method not in ('GET','HEAD'):
        origin = request.headers.get('origin')
        if origin and origin not in ('http://127.0.0.1:8765','http://localhost:8765'):
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'Cross-origin writes are disabled'}, status_code=403)
    return await call_next(request)

class Import(BaseModel):
    path: str = ''
    password: str = ''
    register_only: bool = False

class Extraction(BaseModel):
    document_id: str
    model: str
    stop_on_output_limit: bool = True

class SemanticIndexRequest(BaseModel):
    model: str

class EvidenceSearchRequest(BaseModel):
    query: str
    filters: dict = {}
    mode: str = 'hybrid'
    limit: int = 20

class SourceRequest(BaseModel):
    document_id: str
    pages: list[int]

class TimelineRequest(BaseModel):
    start: str | None = None
    end: str | None = None
    document_id: str | None = None
    limit: int = 500

class CriteriaRequest(BaseModel):
    topic: str

class BatchRequest(BaseModel):
    folder: str
    ocr_model: str = ''
    evidence_model: str = ''
    medgemma_model: str = ''
    recursive: bool = False
    stop_on_output_limit: bool = True

class Decision(BaseModel):
    status: str
    notes: str = ''

class PageIgnore(BaseModel):
    ignored: bool

class TranscriptionEdit(BaseModel):
    text: str

def validate_ocr_model(model):
    normalized=model.casefold()
    if 'medgemma' in normalized and 'text' in normalized:
        raise HTTPException(400,'MedGemma text is an evidence-extraction model, not an OCR model. Select LightOnOCR in the OCR model field and keep MedGemma in its separate evidence field.')

def validate_loaded_model(model,stage):
    try:
        models=core.list_models()
    except Exception as exc:
        raise HTTPException(503,'Cannot verify models in LM Studio. Start its local server and retry.') from exc
    if not any(item.casefold()==model.casefold() or item.casefold().endswith('/'+model.casefold()) for item in models):
        raise HTTPException(409,f'The selected {stage} model is not locally available in LM Studio. Available models: {", ".join(models) or "none"}.')

def completed_stage_pages(db, document_id, kind):
    if kind=='ocr':
        return db.execute("SELECT count(DISTINCT t.page_id) FROM transcriptions t JOIN pages p ON p.id=t.page_id WHERE t.status='complete' AND t.model!='manual-edit' AND p.document_id=? AND p.ignored=0",(document_id,)).fetchone()[0]
    return db.execute("SELECT count(DISTINCT r.page_id) FROM runs r JOIN jobs j ON j.id=r.job_id JOIN pages p ON p.id=r.page_id WHERE p.document_id=? AND p.ignored=0 AND (j.kind=? OR (j.kind IS NULL AND ?='evidence')) AND r.status='complete'",(document_id,kind,kind)).fetchone()[0]

def ensure_stage_empty(db, document_id, kind):
    completed=completed_stage_pages(db,document_id,kind)
    if completed:
        label={'ocr':'OCR','evidence':'NuExtract','medgemma':'MedGemma'}[kind]
        raise HTTPException(409,f'{label} already has completed data for {completed} page(s). Delete {label} data before starting a fresh run. Incomplete jobs resume automatically when you press the stage button again.')

def incomplete_stage_job(db, document_id, kind):
    """Return the latest unfinished LM Studio job for this stage, if any."""
    kind_clause="(kind=? OR (kind IS NULL AND ?='evidence'))"
    row=db.execute(f'''SELECT id,model,status,stop_on_output_limit FROM jobs
        WHERE document_id=? AND provider='lmstudio' AND {kind_clause}
          AND status IN ('needs_attention','interrupted','failed')
        ORDER BY id DESC LIMIT 1''',(document_id,kind,kind)).fetchone()
    return row

@app.get('/api/transcriptions/{page_id}')
def transcriptions(page_id: str):
    with core.connect() as db:
        latest=db.execute("SELECT id FROM transcriptions WHERE page_id=? AND status='complete' ORDER BY id DESC LIMIT 1",(page_id,)).fetchone()
        return {'versions':[dict(r) for r in db.execute('SELECT * FROM transcriptions WHERE page_id=? ORDER BY id DESC',(page_id,))], 'latest_transcription_id':latest['id'] if latest else None}

@app.post('/api/pages/{page_id}/ignore')
def set_page_ignored(page_id: str, data: PageIgnore):
    if not lock.acquire(False):
        raise HTTPException(409,'Wait for the current processing task to finish before changing page ignore state.')
    try:
        with core.connect() as db:
            page=db.execute('SELECT id,document_id,number FROM pages WHERE id=?',(page_id,)).fetchone()
            if not page:
                raise HTTPException(404,'Page not found.')
            db.execute('UPDATE pages SET ignored=? WHERE id=?',(int(data.ignored),page_id))
            if data.ignored:
                core.clear_page_processing_data(db,page_id)
            else:
                db.execute('''INSERT INTO page_search(page_id,text)
                    SELECT p.id,t.text FROM pages p JOIN transcriptions t ON t.id=(
                        SELECT x.id FROM transcriptions x WHERE x.page_id=p.id AND x.status='complete' ORDER BY x.id DESC LIMIT 1)
                    WHERE p.id=? AND p.ignored=0 AND t.text IS NOT NULL AND trim(t.text)!='' ''',(page_id,))
        return {'page_id':page_id,'ignored':data.ignored}
    finally:
        lock.release()

@app.post('/api/transcriptions/{page_id}/edit')
def edit_transcription(page_id: str, data: TranscriptionEdit):
    if not data.text.strip():
        raise HTTPException(400,'OCR text cannot be empty. Use Ignore page for a blank or unusable page.')
    if len(data.text)>200000:
        raise HTTPException(413,'OCR text is too long to save as one page transcription.')
    if not lock.acquire(False):
        raise HTTPException(409,'Wait for the current processing task to finish before editing OCR text.')
    try:
        with core.connect() as db:
            page=db.execute('SELECT id,ignored FROM pages WHERE id=?',(page_id,)).fetchone()
            if not page:
                raise HTTPException(404,'Page not found.')
            if page['ignored']:
                raise HTTPException(409,'Include this page before editing its OCR transcription.')
            transcription_id=db.execute("INSERT INTO transcriptions(page_id,job_id,model,text,status,error,image_hash,created) VALUES(?,NULL,'manual-edit',?,'complete',NULL,NULL,?)",(page_id,data.text,core.now())).lastrowid
            core.rebuild_page_search(db)
        return {'saved':True,'transcription_id':transcription_id}
    finally:
        lock.release()

@app.post('/api/ocr')
def start_ocr(data: Extraction):
    if not data.model.strip():
        raise HTTPException(400,'Select the LightOnOCR model identifier first.')
    validate_ocr_model(data.model)
    validate_loaded_model(data.model,'OCR')
    if not lock.acquire(False):
        raise HTTPException(409,'Another processing task is running.')
    try:
        with core.connect() as db:
            if not db.execute('SELECT id FROM pages WHERE document_id=?',(data.document_id,)).fetchone():
                raise HTTPException(400,'Prepare pages before running OCR.')
            ensure_stage_empty(db,data.document_id,'ocr')
            jid=db.execute('INSERT INTO jobs(document_id,status,model,created,provider,kind) VALUES(?,?,?,?,?,?)',(data.document_id,'queued',data.model,core.now(),'lmstudio','ocr')).lastrowid
    except Exception:
        lock.release()
        raise
    threading.Thread(target=start_job,args=(jid,),daemon=True).start()
    return {'job_id':jid}

@app.post('/api/select-folder')
def select_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk()
        root.withdraw()
        root.attributes('-topmost',True)
        folder=filedialog.askdirectory(parent=root,title='Choose a folder of PDFs and image files',mustexist=True)
        root.destroy()
        return {'folder':folder}
    except Exception as exc:
        raise HTTPException(503,f'Could not open the Windows folder picker: {exc}')

@app.post('/api/batches')
def start_batch(data: BatchRequest):
    if not data.ocr_model.strip() or not data.evidence_model.strip() or 'cloud' in data.ocr_model.lower() or 'cloud' in data.evidence_model.lower() or 'cloud' in data.medgemma_model.lower():
        raise HTTPException(400,'Choose installed local OCR and evidence models first.')
    validate_ocr_model(data.ocr_model)
    validate_loaded_model(data.ocr_model,'OCR')
    validate_loaded_model(data.evidence_model,'NuExtract')
    if data.medgemma_model.strip():
        validate_loaded_model(data.medgemma_model,'MedGemma')
    folder=Path(data.folder).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(400,'Choose an existing folder.')
    iterator=folder.rglob('*') if data.recursive else folder.iterdir()
    files=sorted((p for p in iterator if p.is_file() and p.suffix.lower() in (core.IMAGE_EXTENSIONS | {'.pdf'})),key=lambda p:str(p).casefold())
    if not files:
        raise HTTPException(400,'No PDF, PNG, JPEG, TIFF or BMP files found in that folder.')
    if not lock.acquire(False):
        raise HTTPException(409,'Another processing task is running.')
    try:
        with core.connect() as db:
            batch_id=db.execute('INSERT INTO batches(folder,status,ocr_model,evidence_model,medgemma_model,created,updated,total,completed,stop_on_output_limit) VALUES(?,?,?,?,?,?,?,?,0,?)',
                (str(folder),'queued',data.ocr_model,data.evidence_model,data.medgemma_model.strip() or None,core.now(),core.now(),len(files),int(data.stop_on_output_limit))).lastrowid
            db.executemany('INSERT INTO batch_items(batch_id,path,name,status) VALUES(?,?,?,?)',
                [(batch_id,str(p),p.name,'queued') for p in files])
        threading.Thread(target=run_batch,args=(batch_id,),daemon=True).start()
    except Exception:
        lock.release()
        raise
    return {'batch_id':batch_id,'total':len(files)}

@app.get('/api/batches/{batch_id}')
def batch_status(batch_id: int):
    with core.connect() as db:
        batch=db.execute('SELECT * FROM batches WHERE id=?',(batch_id,)).fetchone()
        if not batch:
            raise HTTPException(404,'Batch not found.')
        items=[dict(r) for r in db.execute('''SELECT i.*,
            ocr.max_input_tokens AS ocr_peak_input,ocr.max_output_tokens AS ocr_peak_output,ocr.context_length AS ocr_context_length,ocr.output_token_limit AS ocr_output_limit,ocr.current_page AS ocr_current_page,ocr.completed_pages AS ocr_done_pages,ocr.total_pages AS ocr_total_pages,
            ev.max_input_tokens AS evidence_peak_input,ev.max_output_tokens AS evidence_peak_output,ev.context_length AS evidence_context_length,ev.output_token_limit AS evidence_output_limit,ev.current_page AS evidence_current_page,ev.completed_pages AS evidence_done_pages,ev.total_pages AS evidence_total_pages,
            med.max_input_tokens AS medgemma_peak_input,med.max_output_tokens AS medgemma_peak_output,med.context_length AS medgemma_context_length,med.output_token_limit AS medgemma_output_limit,med.current_page AS medgemma_current_page,med.completed_pages AS medgemma_done_pages,med.total_pages AS medgemma_total_pages
            FROM batch_items i
            LEFT JOIN jobs ocr ON ocr.id=i.ocr_job_id
            LEFT JOIN jobs ev ON ev.id=i.evidence_job_id
            LEFT JOIN jobs med ON med.id=i.medgemma_job_id
            WHERE i.batch_id=? ORDER BY i.id''',(batch_id,))]
        return {'batch':dict(batch),'items':items}

@app.get('/api/batches')
def latest_batch():
    with core.connect() as db:
        row=db.execute('SELECT id FROM batches ORDER BY id DESC LIMIT 1').fetchone()
    return batch_status(row['id']) if row else {}

class DeleteDocument(BaseModel):
    confirm_name: str

@app.post('/api/documents/{document_id}/delete')
def delete_document(document_id: str, data: DeleteDocument):
    if not lock.acquire(False):
        raise HTTPException(409,'Wait for the current processing task to finish before deleting a file.')
    try:
        with core.connect() as db:
            row = db.execute('SELECT name FROM documents WHERE id=?',(document_id,)).fetchone()
        if not row:
            raise HTTPException(404,'Document not found.')
        if data.confirm_name != row['name']:
            raise HTTPException(400,'The confirmation filename does not match.')
        core.delete_document(document_id)
        return {'deleted':document_id}
    except (ValueError,OSError) as exc:
        raise HTTPException(400,str(exc))
    finally:
        lock.release()

class ClearStage(BaseModel):
    stage: str

class SelectedWorkflow(BaseModel):
    stage: str
    document_ids: list[str]
    model: str = ''
    stop_on_output_limit: bool = True

@app.post('/api/documents/{document_id}/clear-stage')
def clear_stage(document_id: str, data: ClearStage):
    if not lock.acquire(False):
        raise HTTPException(409,'Wait for the current processing task to finish before clearing stage data.')
    try:
        return core.clear_stage_data(document_id,data.stage)
    except ValueError as exc:
        raise HTTPException(400,str(exc))
    finally:
        lock.release()

def run_selected_stage(jobs):
    try:
        for jid,kind in jobs:
            try:
                (core.ocr if kind=='ocr' else core.extract)(jid)
            except Exception as exc:
                with core.connect() as db:
                    core.finish_job(db,jid,'needs_attention',str(exc))
    finally:
        lock.release()

@app.post('/api/workflow-selected')
def workflow_selected(data: SelectedWorkflow):
    if data.stage not in ('ocr','evidence','medgemma'):
        raise HTTPException(400,'Choose OCR, NuExtract or MedGemma.')
    if not data.document_ids or len(set(data.document_ids))!=len(data.document_ids):
        raise HTTPException(400,'Select at least one file.')
    kind=data.stage
    if not data.model.strip() or 'cloud' in data.model.casefold():
        raise HTTPException(400,'Choose an installed local model for this stage.')
    if kind=='ocr':
        validate_ocr_model(data.model)
    validate_loaded_model(data.model,{'ocr':'OCR','evidence':'NuExtract','medgemma':'MedGemma'}[kind])
    if not lock.acquire(False):
        raise HTTPException(409,'Another processing task is running.')
    try:
        with core.connect() as db:
            ids=set(data.document_ids)
            existing={r['id'] for r in db.execute('SELECT id FROM documents WHERE id IN ('+','.join('?' for _ in ids)+')',tuple(ids))}
            if existing!=ids:
                raise HTTPException(404,'One or more selected files no longer exist.')
            if kind in ('evidence','medgemma'):
                missing=db.execute('SELECT count(*) FROM pages p WHERE p.document_id IN ('+','.join('?' for _ in ids)+') AND p.ignored=0 AND NOT EXISTS (SELECT 1 FROM transcriptions t WHERE t.page_id=p.id AND t.status=\'complete\')',tuple(ids)).fetchone()[0]
                if missing:
                    raise HTTPException(409,f'Run OCR successfully on every selected page first; {missing} page(s) still need OCR.')
            plans=[]
            for document_id in data.document_ids:
                previous=incomplete_stage_job(db,document_id,kind)
                if previous:
                    if previous['model'].casefold()!=data.model.casefold():
                        raise HTTPException(409,f'An incomplete {kind} run for this file uses model “{previous["model"]}”. Select that model to resume it, or delete this stage data before starting with a different model.')
                    plans.append((document_id,previous['id']))
                else:
                    ensure_stage_empty(db,document_id,kind)
                    plans.append((document_id,None))
            jobs=[]
            for document_id,previous_id in plans:
                if previous_id is not None:
                    jid=previous_id
                    db.execute("UPDATE jobs SET stop_on_output_limit=?,status='queued' WHERE id=?",(int(data.stop_on_output_limit),jid))
                else:
                    jid=db.execute('INSERT INTO jobs(document_id,status,model,created,provider,kind,stop_on_output_limit) VALUES(?,?,?,?,?,?,?)',(document_id,'queued',data.model,core.now(),'lmstudio',kind,int(data.stop_on_output_limit))).lastrowid
                jobs.append((jid,kind))
        threading.Thread(target=run_selected_stage,args=(jobs,),daemon=True).start()
        return {'job_ids':[jid for jid,_ in jobs]}
    except Exception:
        lock.release()
        raise

@app.post('/api/documents/{document_id}/pages')
def resume_pages(document_id: str, data: Import):
    with core.connect() as db:
        row = db.execute('SELECT source_ext FROM documents WHERE id=?',(document_id,)).fetchone()
    if not row:
        raise HTTPException(404,'Document not found.')
    # Use the preserved import even if the external source moved or changed.
    ext=row['source_ext'] if row['source_ext'] in (core.IMAGE_EXTENSIONS | {'.pdf'}) else '.pdf'
    return import_pdf(Import(path=str(core.ROOT / 'originals' / (document_id+ext)),password=data.password))

@app.get('/api/dataset')
def dataset():
    with core.connect() as db:
        documents=[dict(r) for r in db.execute('''SELECT d.*,
            (SELECT count(*) FROM pages WHERE document_id=d.id AND status='ready' AND ignored=0) AS ready,
            (SELECT count(*) FROM pages WHERE document_id=d.id AND status='needs_ocr' AND ignored=0) AS needs_ocr,
            (SELECT count(*) FROM pages WHERE document_id=d.id AND status='failed' AND ignored=0) AS failed,
            (SELECT count(DISTINCT t.page_id) FROM transcriptions t JOIN pages p ON t.page_id=p.id WHERE p.document_id=d.id AND t.status='complete' AND p.ignored=0) AS transcribed,
            (SELECT count(*) FROM evidence e JOIN pages p ON e.page_id=p.id WHERE p.document_id=d.id) AS evidence_count,
            (SELECT count(*) FROM evidence e JOIN pages p ON e.page_id=p.id WHERE p.document_id=d.id AND e.status='unreviewed') AS unreviewed
            ,(SELECT count(DISTINCT r.page_id) FROM runs r JOIN jobs j ON j.id=r.job_id JOIN pages p ON p.id=r.page_id WHERE p.document_id=d.id AND p.ignored=0 AND (j.kind='evidence' OR j.kind IS NULL) AND r.status='complete') AS evidence_data_pages
            ,(SELECT count(DISTINCT r.page_id) FROM runs r JOIN jobs j ON j.id=r.job_id JOIN pages p ON p.id=r.page_id WHERE p.document_id=d.id AND p.ignored=0 AND j.kind='medgemma' AND r.status='complete') AS medgemma_data_pages
            ,(SELECT j.id FROM jobs j WHERE j.document_id=d.id AND j.kind='ocr' ORDER BY j.id DESC LIMIT 1) AS ocr_job_id
            ,(SELECT j.status FROM jobs j WHERE j.document_id=d.id AND j.kind='ocr' ORDER BY j.id DESC LIMIT 1) AS ocr_status
            ,(SELECT j.current_page FROM jobs j WHERE j.document_id=d.id AND j.kind='ocr' ORDER BY j.id DESC LIMIT 1) AS ocr_current_page
            ,(SELECT j.completed_pages FROM jobs j WHERE j.document_id=d.id AND j.kind='ocr' ORDER BY j.id DESC LIMIT 1) AS ocr_completed_pages
            ,(SELECT j.total_pages FROM jobs j WHERE j.document_id=d.id AND j.kind='ocr' ORDER BY j.id DESC LIMIT 1) AS ocr_total_pages
            ,(SELECT j.last_completed_page FROM jobs j WHERE j.document_id=d.id AND j.kind='ocr' ORDER BY j.id DESC LIMIT 1) AS ocr_last_completed_page
            ,(SELECT j.started_at FROM jobs j WHERE j.document_id=d.id AND j.kind='ocr' ORDER BY j.id DESC LIMIT 1) AS ocr_started_at
            ,(SELECT j.finished_at FROM jobs j WHERE j.document_id=d.id AND j.kind='ocr' ORDER BY j.id DESC LIMIT 1) AS ocr_finished_at
            ,(SELECT j.id FROM jobs j WHERE j.document_id=d.id AND j.kind='evidence' ORDER BY j.id DESC LIMIT 1) AS evidence_job_id
            ,(SELECT j.status FROM jobs j WHERE j.document_id=d.id AND j.kind='evidence' ORDER BY j.id DESC LIMIT 1) AS evidence_status
            ,(SELECT j.current_page FROM jobs j WHERE j.document_id=d.id AND j.kind='evidence' ORDER BY j.id DESC LIMIT 1) AS evidence_current_page
            ,(SELECT j.completed_pages FROM jobs j WHERE j.document_id=d.id AND j.kind='evidence' ORDER BY j.id DESC LIMIT 1) AS evidence_completed_pages
            ,(SELECT j.total_pages FROM jobs j WHERE j.document_id=d.id AND j.kind='evidence' ORDER BY j.id DESC LIMIT 1) AS evidence_total_pages
            ,(SELECT j.last_completed_page FROM jobs j WHERE j.document_id=d.id AND j.kind='evidence' ORDER BY j.id DESC LIMIT 1) AS evidence_last_completed_page
            ,(SELECT j.started_at FROM jobs j WHERE j.document_id=d.id AND j.kind='evidence' ORDER BY j.id DESC LIMIT 1) AS evidence_started_at
            ,(SELECT j.finished_at FROM jobs j WHERE j.document_id=d.id AND j.kind='evidence' ORDER BY j.id DESC LIMIT 1) AS evidence_finished_at
            ,(SELECT j.id FROM jobs j WHERE j.document_id=d.id AND j.kind='medgemma' ORDER BY j.id DESC LIMIT 1) AS medgemma_job_id
            ,(SELECT j.status FROM jobs j WHERE j.document_id=d.id AND j.kind='medgemma' ORDER BY j.id DESC LIMIT 1) AS medgemma_status
            ,(SELECT j.current_page FROM jobs j WHERE j.document_id=d.id AND j.kind='medgemma' ORDER BY j.id DESC LIMIT 1) AS medgemma_current_page
            ,(SELECT j.completed_pages FROM jobs j WHERE j.document_id=d.id AND j.kind='medgemma' ORDER BY j.id DESC LIMIT 1) AS medgemma_completed_pages
            ,(SELECT j.total_pages FROM jobs j WHERE j.document_id=d.id AND j.kind='medgemma' ORDER BY j.id DESC LIMIT 1) AS medgemma_total_pages
            ,(SELECT j.last_completed_page FROM jobs j WHERE j.document_id=d.id AND j.kind='medgemma' ORDER BY j.id DESC LIMIT 1) AS medgemma_last_completed_page
            ,(SELECT j.started_at FROM jobs j WHERE j.document_id=d.id AND j.kind='medgemma' ORDER BY j.id DESC LIMIT 1) AS medgemma_started_at
            ,(SELECT j.finished_at FROM jobs j WHERE j.document_id=d.id AND j.kind='medgemma' ORDER BY j.id DESC LIMIT 1) AS medgemma_finished_at
            FROM documents d ORDER BY name''')]
        for document in documents:
            for kind,key in (('ocr','ocr'),('evidence','evidence'),('medgemma','medgemma')):
                document[key+'_runtime_seconds']=core.document_stage_runtime(db,document['id'],kind)
        return documents

@app.get('/api/documents/{document_id}/messages')
def document_messages(document_id: str):
    with core.connect() as db:
        if not db.execute('SELECT id FROM documents WHERE id=?',(document_id,)).fetchone():
            raise HTTPException(404,'Document not found.')
        rows=db.execute('''SELECT b.id AS batch_id,b.status AS batch_status,b.folder,b.created,b.started_at,b.finished_at,
                i.status AS file_status,i.error,i.pages,i.transcribed,i.candidates,i.medgemma_candidates
            FROM batch_items i JOIN batches b ON b.id=i.batch_id
            WHERE i.document_id=? ORDER BY b.id DESC,i.id DESC''',(document_id,)).fetchall()
        return [dict(row) for row in rows]

@app.get('/')
def home():
    return FileResponse(Path(__file__).parent / 'index.html',headers={'Cache-Control':'no-store'})

@app.post('/api/import')
def import_pdf(data: Import):
    if not lock.acquire(False):
        raise HTTPException(409,'Another processing task is running.')
    try:
        return {'id':core.ingest(data.path,data.password,data.register_only)}
    except Exception as exc:
        raise HTTPException(400,str(exc))
    finally:
        lock.release()

@app.post('/api/import-folder')
def import_folder(data: BatchRequest):
    folder=Path(data.folder).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(400,'Choose an existing folder.')
    iterator=folder.rglob('*') if data.recursive else folder.iterdir()
    files=sorted((p for p in iterator if p.is_file() and p.suffix.lower() in (core.IMAGE_EXTENSIONS | {'.pdf'})),key=lambda p:str(p).casefold())
    if not files:
        raise HTTPException(400,'No PDF, PNG, JPEG, TIFF or BMP files found in that folder.')
    if not lock.acquire(False):
        raise HTTPException(409,'Another processing task is running.')
    try:
        imported=[]
        for path in files:
            imported.append(core.ingest(path,register_only=True))
        return {'imported':len(imported),'document_ids':imported}
    except Exception as exc:
        raise HTTPException(400,str(exc))
    finally:
        lock.release()

@app.get('/api/state')
def state():
    with core.connect() as db:
        jobs=db.execute('''SELECT j.*,
            CASE WHEN j.kind='ocr' THEN (SELECT count(*) FROM transcriptions t WHERE t.job_id=j.id AND t.status='failed')
                 ELSE (SELECT count(*) FROM runs r WHERE r.job_id=j.id AND r.status='failed') END AS retryable
            FROM jobs j ORDER BY j.id DESC''').fetchall()
        return {'documents':[dict(r) for r in db.execute('SELECT * FROM documents')], 'jobs':[dict(r) for r in jobs]}

@app.get('/api/models')
def models():
    try:
        details=core.list_model_details()
        models=[item.get('key') for item in details if item.get('key')]
        types={item['key']:item.get('type') for item in details if item.get('key')}
        return {'models':models,'types':types}
    except urllib.error.HTTPError as exc:
        if exc.code in (401,403):
            raise HTTPException(503,'LM Studio requires authentication. Set LM_API_TOKEN in the reviewer server environment and restart the reviewer.')
        raise HTTPException(503,'LM Studio returned an error. Check its server logs.')
    except Exception:
        raise HTTPException(503,'LM Studio is not reachable. Start its local server on port 1234 and load a model.')

@app.get('/api/model-status')
def model_status():
    try:
        current=core.list_models()
        lm_startup_status.update(available=True,models=current,error=None)
    except Exception as exc:
        lm_startup_status.update(available=False,models=[],error=str(exc))
    return dict(lm_startup_status)

@app.get('/api/model-contexts')
def model_contexts(ocr_model: str = '', evidence_model: str = '', medgemma_model: str = ''):
    try:
        contexts=core.list_model_contexts()
    except urllib.error.HTTPError as exc:
        raise HTTPException(503,'LM Studio model context settings are unavailable. Check the local server and API token.') from exc
    except Exception:
        raise HTTPException(503,'LM Studio model context settings are unavailable. Start its local server.')
    def budget(model,output_limit,ocr=False):
        entry=contexts.get(model) or next((value for key,value in contexts.items() if model and (key.casefold()==model.casefold() or key.casefold().endswith('/'+model.casefold()))),None)
        context=entry.get('context_length') if entry else None
        if ocr:
            context=16384
        return {'model':model or None,'loaded':bool(entry and entry.get('loaded')) if not ocr else True,'context_length':context,'max_context_length':entry.get('max_context_length') if entry else None,'max_input_tokens':max(context-output_limit,0) if context and output_limit is not None else None,'max_output_tokens':output_limit,'output_limited_by_context':output_limit is None}
    return {'stages':{'ocr':budget(ocr_model,8192,True),'evidence':budget(evidence_model,core.EVIDENCE_OUTPUT_TOKEN_LIMIT),'medgemma':budget(medgemma_model,core.MEDGEMMA_OUTPUT_TOKEN_LIMIT)}}

def run_semantic_index(job_id):
    try:
        core.build_semantic_index(job_id)
    except Exception as exc:
        with core.connect() as db:
            db.execute("UPDATE semantic_jobs SET status='failed',error=?,updated=? WHERE id=?",(str(exc),core.now(),job_id))
    finally:
        lock.release()

@app.get('/api/semantic-index')
def semantic_index_status():
    with core.connect() as db:
        active=db.execute("SELECT value FROM semantic_settings WHERE key='active_model'").fetchone()
        pages=db.execute("SELECT count(DISTINCT p.id) FROM pages p JOIN transcriptions t ON t.id=(SELECT x.id FROM transcriptions x WHERE x.page_id=p.id AND x.status='complete' ORDER BY x.id DESC LIMIT 1) WHERE t.text IS NOT NULL AND trim(t.text)!=''").fetchone()[0]
        indexed=db.execute("SELECT count(DISTINCT si.page_id) FROM semantic_page_index si JOIN transcriptions t ON t.id=si.transcription_id WHERE si.model=coalesce((SELECT value FROM semantic_settings WHERE key='active_model'),'') AND t.id=(SELECT max(x.id) FROM transcriptions x WHERE x.page_id=si.page_id AND x.status='complete')").fetchone()[0]
        job=db.execute('SELECT * FROM semantic_jobs ORDER BY id DESC LIMIT 1').fetchone()
        return {'active_model':active['value'] if active else None,'indexed_pages':indexed,'eligible_pages':pages,'latest_job':dict(job) if job else None}

@app.post('/api/semantic-index')
def start_semantic_index(data: SemanticIndexRequest):
    if not data.model.strip() or 'cloud' in data.model.casefold():
        raise HTTPException(400,'Choose a local embedding model loaded in LM Studio.')
    if not lock.acquire(False):
        raise HTTPException(409,'Another processing task is running.')
    try:
        with core.connect() as db:
            total=db.execute("SELECT count(*) FROM pages p JOIN transcriptions t ON t.id=(SELECT x.id FROM transcriptions x WHERE x.page_id=p.id AND x.status='complete' ORDER BY x.id DESC LIMIT 1) WHERE t.text IS NOT NULL AND trim(t.text)!=''").fetchone()[0]
            if not total:
                raise HTTPException(409,'Run OCR before building a semantic index.')
            job_id=db.execute("INSERT INTO semantic_jobs(status,model,created,updated,total,completed) VALUES('queued',?,?,?,?,0)",(data.model,core.now(),core.now(),total)).lastrowid
        threading.Thread(target=run_semantic_index,args=(job_id,),daemon=True).start()
        return {'job_id':job_id,'total':total}
    except Exception:
        lock.release()
        raise

@app.get('/api/semantic-index/jobs/{job_id}')
def semantic_index_job(job_id: int):
    with core.connect() as db:
        job=db.execute('SELECT * FROM semantic_jobs WHERE id=?',(job_id,)).fetchone()
        if not job:
            raise HTTPException(404,'Semantic indexing job not found.')
        return dict(job)

@app.get('/api/pages/{document_id}')
def pages(document_id: str):
    with core.connect() as db:
        rows=db.execute('''SELECT p.*,t.id AS transcription_id,t.model AS ocr_model,t.text AS ocr_text,t.created AS transcribed_at,
            (SELECT f.error FROM transcriptions f WHERE f.page_id=p.id AND f.status='failed' ORDER BY f.id DESC LIMIT 1) AS ocr_error,
            CASE WHEN p.ignored=1 THEN 'ignored'
                 WHEN t.id IS NOT NULL THEN 'complete'
                 WHEN EXISTS(SELECT 1 FROM jobs j WHERE j.document_id=p.document_id AND j.kind='ocr' AND j.status='running' AND j.current_page=p.number) THEN 'running'
                 WHEN EXISTS(SELECT 1 FROM transcriptions f WHERE f.page_id=p.id AND f.status='failed' AND f.error LIKE 'Suspected Blank Page%') THEN 'suspected_blank'
                 WHEN EXISTS(SELECT 1 FROM transcriptions f WHERE f.page_id=p.id AND f.status='failed') THEN 'failed'
                 WHEN p.status='failed' THEN 'failed'
                 ELSE 'pending' END AS ocr_progress,
            CASE WHEN p.ignored=1 THEN 'ignored'
                 WHEN EXISTS(SELECT 1 FROM runs r JOIN jobs j ON j.id=r.job_id WHERE r.page_id=p.id AND r.status='complete' AND (j.kind='evidence' OR j.kind IS NULL)) THEN 'complete'
                 WHEN EXISTS(SELECT 1 FROM jobs j WHERE j.document_id=p.document_id AND (j.kind='evidence' OR j.kind IS NULL) AND j.status='running' AND j.current_page=p.number) THEN 'running'
                 WHEN EXISTS(SELECT 1 FROM runs r JOIN jobs j ON j.id=r.job_id WHERE r.page_id=p.id AND r.status='failed' AND (j.kind='evidence' OR j.kind IS NULL)) THEN 'failed'
                 ELSE 'pending' END AS nuextract_progress,
            CASE WHEN p.ignored=1 THEN 'ignored'
                 WHEN EXISTS(SELECT 1 FROM runs r JOIN jobs j ON j.id=r.job_id WHERE r.page_id=p.id AND j.kind='medgemma' AND r.status='complete') THEN 'complete'
                 WHEN EXISTS(SELECT 1 FROM jobs j WHERE j.document_id=p.document_id AND j.kind='medgemma' AND j.status='running' AND j.current_page=p.number) THEN 'running'
                 WHEN EXISTS(SELECT 1 FROM runs r JOIN jobs j ON j.id=r.job_id WHERE r.page_id=p.id AND j.kind='medgemma' AND r.status='failed') THEN 'failed'
                 ELSE 'pending' END AS medgemma_progress
            FROM pages p LEFT JOIN transcriptions t ON t.id=(SELECT x.id FROM transcriptions x WHERE x.page_id=p.id AND x.status='complete' ORDER BY x.id DESC LIMIT 1)
                WHERE p.document_id=? ORDER BY p.number''',(document_id,)).fetchall()
        return [dict(row) for row in rows]

@app.get('/api/image/{page_id}')
def image(page_id: str):
    with core.connect() as db:
        row = db.execute('SELECT image FROM pages WHERE id=?',(page_id,)).fetchone()
    if not row or not Path(row['image']).is_file():
        raise HTTPException(404)
    return FileResponse(row['image'])

@app.get('/api/evidence/{page_id}')
def evidence(page_id: str):
    with core.connect() as db:
        return [dict(r) for r in db.execute('''SELECT e.*,r.model AS source_model,j.kind AS source_stage
            FROM evidence e JOIN runs r ON r.id=e.run_id JOIN jobs j ON j.id=r.job_id
            WHERE e.page_id=? ORDER BY e.id''',(page_id,))]

@app.post('/api/evidence/{eid}')
def review(eid: int, decision: Decision):
    if decision.status not in ('confirmed','rejected','needs_investigation','unreviewed'):
        raise HTTPException(400,'Invalid review status')
    with core.connect() as db:
        if not db.execute('SELECT id FROM evidence WHERE id=?',(eid,)).fetchone():
            raise HTTPException(404)
        db.execute('UPDATE evidence SET status=?,notes=? WHERE id=?',(decision.status,decision.notes,eid))
        db.execute('INSERT INTO decisions(evidence_id,status,notes,created) VALUES(?,?,?,?)',(eid,decision.status,decision.notes,core.now()))
    return {'saved':True}

def start_job(jid):
    try:
        with core.connect() as db:
            job=db.execute('SELECT kind FROM jobs WHERE id=?',(jid,)).fetchone()
        (core.ocr if job['kind']=='ocr' else core.extract)(jid)
    except Exception as exc:
        with core.connect() as db:
            core.finish_job(db,jid,'needs_attention',str(exc))
    finally:
        lock.release()

def _append_batch_error(item_id, message):
    with core.connect() as db:
        row=db.execute('SELECT error FROM batch_items WHERE id=?',(item_id,)).fetchone()
        old=row['error'] if row else None
        db.execute('UPDATE batch_items SET error=? WHERE id=?',((old+'\n' if old else '')+message,item_id))

def _batch_extraction_result(job_id, item_id, label, candidates_column):
    with core.connect() as db:
        result=db.execute('SELECT status,error FROM jobs WHERE id=?',(job_id,)).fetchone()
        candidates=db.execute('SELECT count(*) FROM evidence WHERE run_id IN (SELECT id FROM runs WHERE job_id=?)',(job_id,)).fetchone()[0]
        failures=db.execute("SELECT p.number,r.error FROM runs r JOIN pages p ON p.id=r.page_id WHERE r.job_id=? AND r.status='failed' ORDER BY p.number",(job_id,)).fetchall()
        warnings=db.execute("SELECT p.number,count(*) AS count FROM evidence e JOIN pages p ON p.id=e.page_id WHERE e.run_id IN (SELECT id FROM runs WHERE job_id=?) AND e.quote_match=0 GROUP BY p.number ORDER BY p.number",(job_id,)).fetchall()
        db.execute(f'UPDATE batch_items SET {candidates_column}=? WHERE id=?',(candidates,item_id))
    if result['status']!='complete':
        for row in failures:
            _append_batch_error(item_id,f'{label} page {row["number"]}: {row["error"]}')
        for row in warnings:
            _append_batch_error(item_id,f'{label} page {row["number"]}: {row["count"]} quotation mismatch(es) flagged; candidates were saved for review.')
        if not failures and not warnings:
            _append_batch_error(item_id,f'{label} extraction: '+(result['error'] or 'some pages need attention'))

def run_batch(batch_id):
    try:
        with core.connect() as db:
            batch=db.execute('SELECT * FROM batches WHERE id=?',(batch_id,)).fetchone()
            items=db.execute('SELECT * FROM batch_items WHERE batch_id=? ORDER BY id',(batch_id,)).fetchall()
            batch_start=core.now()
            db.execute("UPDATE batches SET status='running',started_at=COALESCE(started_at,?),finished_at=NULL,updated=? WHERE id=?",(batch_start,batch_start,batch_id))

        # Prepare every source before calling models, then process each model in
        # a batch-wide phase to avoid needless LM Studio model load/unload cycles.
        for item in items:
            try:
                with core.connect() as db:
                    db.execute("UPDATE batch_items SET status='preparing' WHERE id=?",(item['id'],))
                document_id=core.ingest(item['path'])
                with core.connect() as db:
                    pages=db.execute('SELECT count(*) FROM pages WHERE document_id=?',(document_id,)).fetchone()[0]
                    db.execute("UPDATE batch_items SET status='ocr',document_id=?,pages=? WHERE id=?",(document_id,pages,item['id']))
            except Exception as exc:
                with core.connect() as db:
                    db.execute("UPDATE batch_items SET status='failed',error=? WHERE id=?",(str(exc),item['id']))

        with core.connect() as db:
            ready=db.execute("SELECT * FROM batch_items WHERE batch_id=? AND document_id IS NOT NULL AND status!='failed' ORDER BY id",(batch_id,)).fetchall()

        # OCR phase: keep the OCR model active across all prepared files.
        for item in ready:
            try:
                with core.connect() as db:
                    prior=completed_stage_pages(db,item['document_id'],'ocr')
                    if prior:
                        transcribed=db.execute("SELECT count(DISTINCT t.page_id) FROM transcriptions t JOIN pages p ON p.id=t.page_id WHERE t.status='complete' AND p.document_id=? AND p.ignored=0",(item['document_id'],)).fetchone()[0]
                        db.execute('UPDATE batch_items SET transcribed=? WHERE id=?',(transcribed,item['id']))
                        continue
                with core.connect() as db:
                    job_id=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind,stop_on_output_limit) VALUES(?,?,?,?,?,?,?)",(item['document_id'],'queued',batch['ocr_model'],core.now(),'lmstudio','ocr',batch['stop_on_output_limit'])).lastrowid
                    db.execute('UPDATE batch_items SET ocr_job_id=? WHERE id=?',(job_id,item['id']))
                core.ocr(job_id)
                with core.connect() as db:
                    result=db.execute('SELECT status,error FROM jobs WHERE id=?',(job_id,)).fetchone()
                    transcribed=db.execute("SELECT count(DISTINCT p.id) FROM pages p JOIN transcriptions t ON t.page_id=p.id AND t.status='complete' WHERE p.document_id=? AND p.ignored=0",(item['document_id'],)).fetchone()[0]
                    failures=db.execute("SELECT p.number,t.error FROM transcriptions t JOIN pages p ON p.id=t.page_id WHERE t.job_id=? AND t.status='failed' ORDER BY p.number",(job_id,)).fetchall()
                    db.execute('UPDATE batch_items SET transcribed=? WHERE id=?',(transcribed,item['id']))
                if result['status']!='complete':
                    for row in failures:
                        _append_batch_error(item['id'],f'OCR page {row["number"]}: {row["error"]}')
                    if not failures:
                        _append_batch_error(item['id'],'OCR: '+(result['error'] or 'some pages failed'))
                if not transcribed:
                    _append_batch_error(item['id'],'Evidence extraction skipped: no pages were successfully transcribed.')
            except Exception as exc:
                _append_batch_error(item['id'],'OCR: '+str(exc))

        with core.connect() as db:
            usable=db.execute('SELECT * FROM batch_items WHERE batch_id=? AND document_id IS NOT NULL AND transcribed>0 ORDER BY id',(batch_id,)).fetchall()

        # Primary extraction phase: process every OCR-ready document in turn.
        for item in usable:
            try:
                with core.connect() as db:
                    if completed_stage_pages(db,item['document_id'],'evidence'):
                        candidates=db.execute("SELECT count(*) FROM evidence WHERE run_id IN (SELECT r.id FROM runs r JOIN jobs j ON j.id=r.job_id WHERE j.document_id=? AND j.kind='evidence')",(item['document_id'],)).fetchone()[0]
                        db.execute('UPDATE batch_items SET candidates=? WHERE id=?',(candidates,item['id']))
                        continue
                with core.connect() as db:
                    job_id=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind,stop_on_output_limit) VALUES(?,?,?,?,?,?,?)",(item['document_id'],'queued',batch['evidence_model'],core.now(),'lmstudio','evidence',batch['stop_on_output_limit'])).lastrowid
                    db.execute("UPDATE batch_items SET status='extracting',evidence_job_id=? WHERE id=?",(job_id,item['id']))
                core.extract(job_id)
                _batch_extraction_result(job_id,item['id'],'Evidence','candidates')
            except Exception as exc:
                _append_batch_error(item['id'],'Evidence extraction: '+str(exc))

        # Optional MedGemma phase: all documents are grouped into one pass too.
        if batch['medgemma_model']:
            for item in usable:
                try:
                    with core.connect() as db:
                        if completed_stage_pages(db,item['document_id'],'medgemma'):
                            candidates=db.execute("SELECT count(*) FROM evidence WHERE run_id IN (SELECT r.id FROM runs r JOIN jobs j ON j.id=r.job_id WHERE j.document_id=? AND j.kind='medgemma')",(item['document_id'],)).fetchone()[0]
                            db.execute('UPDATE batch_items SET medgemma_candidates=? WHERE id=?',(candidates,item['id']))
                            continue
                    with core.connect() as db:
                        job_id=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind,stop_on_output_limit) VALUES(?,?,?,?,?,?,?)",(item['document_id'],'queued',batch['medgemma_model'],core.now(),'lmstudio','medgemma',batch['stop_on_output_limit'])).lastrowid
                        db.execute("UPDATE batch_items SET status='medgemma',medgemma_job_id=? WHERE id=?",(job_id,item['id']))
                    core.extract(job_id)
                    _batch_extraction_result(job_id,item['id'],'MedGemma','medgemma_candidates')
                except Exception as exc:
                    _append_batch_error(item['id'],'MedGemma extraction: '+str(exc))

        with core.connect() as db:
            db.execute("UPDATE batch_items SET status=CASE WHEN error IS NULL OR error='' THEN 'complete' ELSE 'needs_attention' END WHERE batch_id=? AND status!='failed'",(batch_id,))
            completed=db.execute("SELECT count(*) FROM batch_items WHERE batch_id=? AND status IN ('complete','needs_attention','failed')",(batch_id,)).fetchone()[0]
            failed=db.execute("SELECT count(*) FROM batch_items WHERE batch_id=? AND status IN ('needs_attention','failed')",(batch_id,)).fetchone()[0]
            batch_finish=core.now()
            db.execute('UPDATE batches SET completed=?,status=?,finished_at=?,updated=? WHERE id=?',(completed,'needs_attention' if failed else 'complete',batch_finish,batch_finish,batch_id))
    except Exception as exc:
        with core.connect() as db:
            batch_finish=core.now()
            db.execute("UPDATE batches SET status='failed',finished_at=?,updated=? WHERE id=?",(batch_finish,batch_finish,batch_id))
            db.execute("UPDATE batch_items SET status='failed',error=COALESCE(error,?) WHERE batch_id=? AND status IN ('queued','preparing','ocr','extracting','medgemma')",(str(exc),batch_id))
    finally:
        lock.release()
@app.post('/api/extract')
def extract(data: Extraction):
    if not data.model.strip() or 'cloud' in data.model.lower():
        raise HTTPException(400,'Specify an installed local model, not a cloud model.')
    validate_loaded_model(data.model,'NuExtract')
    if not lock.acquire(False):
        raise HTTPException(409,'Another processing task is running.')
    with core.connect() as db:
        if not db.execute('SELECT id FROM documents WHERE id=?',(data.document_id,)).fetchone():
            lock.release()
            raise HTTPException(404)
        try:
            ensure_stage_empty(db,data.document_id,'evidence')
        except Exception:
            lock.release()
            raise
        missing=db.execute("""SELECT count(*) FROM pages p WHERE p.document_id=? AND p.ignored=0 AND NOT EXISTS(
            SELECT 1 FROM transcriptions t WHERE t.page_id=p.id AND t.status='complete')""",(data.document_id,)).fetchone()[0]
        if missing:
            lock.release()
            raise HTTPException(409,f'Run OCR successfully for all {missing} remaining pages before extracting evidence. Embedded PDF text is not used.')
        jid = db.execute('INSERT INTO jobs(document_id,status,model,created,provider,kind,stop_on_output_limit) VALUES(?,?,?,?,?,?,?)',(data.document_id,'queued',data.model,core.now(),'lmstudio','evidence',int(data.stop_on_output_limit))).lastrowid
    threading.Thread(target=start_job,args=(jid,),daemon=True).start()
    return {'job_id':jid}

@app.post('/api/extract-medgemma')
def extract_medgemma(data: Extraction):
    if not data.model.strip() or 'cloud' in data.model.lower():
        raise HTTPException(400,'Specify an installed local MedGemma model.')
    validate_loaded_model(data.model,'MedGemma')
    if not lock.acquire(False):
        raise HTTPException(409,'Another processing task is running.')
    try:
        with core.connect() as db:
            if not db.execute('SELECT id FROM documents WHERE id=?',(data.document_id,)).fetchone():
                raise HTTPException(404,'Document not found.')
            ensure_stage_empty(db,data.document_id,'medgemma')
            missing=db.execute("""SELECT count(*) FROM pages p WHERE p.document_id=? AND p.ignored=0 AND NOT EXISTS(
                SELECT 1 FROM transcriptions t WHERE t.page_id=p.id AND t.status='complete')""",(data.document_id,)).fetchone()[0]
            if missing:
                raise HTTPException(409,f'Run OCR successfully for all {missing} remaining pages before MedGemma extraction.')
            jid=db.execute("INSERT INTO jobs(document_id,status,model,created,provider,kind,stop_on_output_limit) VALUES(?,?,?,?,?,?,?)",
                (data.document_id,'queued',data.model,core.now(),'lmstudio','medgemma',int(data.stop_on_output_limit))).lastrowid
        threading.Thread(target=start_job,args=(jid,),daemon=True).start()
        return {'job_id':jid}
    except Exception:
        lock.release()
        raise

@app.post('/api/resume/{jid}')
def resume(jid: int):
    if not lock.acquire(False):
        raise HTTPException(409,'Another processing task is running.')
    with core.connect() as db:
        job = db.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone()
    if not job:
        lock.release()
        raise HTTPException(404)
    if job['provider'] != 'lmstudio':
        lock.release()
        raise HTTPException(409,'This job used Ollama. Start a new extraction with an LM Studio model; previous evidence is preserved.')
    if job['kind']=='evidence':
        with core.connect() as db:
            missing=db.execute("""SELECT count(*) FROM pages p WHERE p.document_id=? AND p.ignored=0 AND NOT EXISTS(
                SELECT 1 FROM transcriptions t WHERE t.page_id=p.id AND t.status='complete')""",(job['document_id'],)).fetchone()[0]
        if missing:
            lock.release()
            raise HTTPException(409,f'Run OCR successfully for {missing} remaining pages before resuming evidence extraction.')
    threading.Thread(target=start_job,args=(jid,),daemon=True).start()
    return {'job_id':jid}

@app.get('/api/runs/{document_id}')
def runs(document_id: str):
    with core.connect() as db:
        return [dict(r) for r in db.execute('SELECT runs.id,runs.page_id,runs.status,runs.error,runs.output,runs.input_tokens,runs.output_tokens,runs.finish_reason,runs.model,j.kind FROM runs JOIN pages ON pages.id=runs.page_id JOIN jobs j ON j.id=runs.job_id WHERE pages.document_id=? AND pages.ignored=0',(document_id,))]

@app.get('/api/search')
def search(q: str, document_id: str | None = None, evidence_status: str | None = None,
          source_stage: str | None = None, mode: str = 'lexical', limit: int = Query(100, ge=1, le=250)):
    if evidence_status not in (None, 'unreviewed', 'confirmed', 'rejected', 'needs_investigation'):
        raise HTTPException(400,'Invalid evidence review status filter.')
    if source_stage not in (None, 'evidence', 'medgemma'):
        raise HTTPException(400,'Invalid evidence source filter.')
    if mode not in ('lexical','semantic','hybrid'):
        raise HTTPException(400,'Search mode must be lexical, semantic or hybrid.')
    if not q.strip():
        return []
    if mode=='semantic':
        try:
            return core.semantic_search(q,document_id,evidence_status,source_stage,limit)
        except ValueError as exc:
            raise HTTPException(409,str(exc))
    if mode=='hybrid':
        try:
            return core.hybrid_search(q,document_id,evidence_status,source_stage,limit)
        except ValueError as exc:
            raise HTTPException(409,str(exc))
    return lexical_search(q,document_id,evidence_status,source_stage,limit)

@app.post('/api/tools/search_evidence')
def tool_search_evidence(data: EvidenceSearchRequest):
    if not data.query.strip():
        raise HTTPException(400,'query must not be empty.')
    if data.mode not in ('lexical','semantic','hybrid'):
        raise HTTPException(400,'mode must be lexical, semantic or hybrid.')
    if not 1<=data.limit<=100:
        raise HTTPException(400,'limit must be between 1 and 100.')
    filters=data.filters or {}
    allowed={'document_id','evidence_status','source_stage'}
    if not isinstance(filters,dict) or set(filters)-allowed:
        raise HTTPException(400,'Supported filters are document_id, evidence_status and source_stage.')
    document_id=filters.get('document_id')
    evidence_status=filters.get('evidence_status')
    source_stage=filters.get('source_stage')
    if evidence_status not in (None,'unreviewed','confirmed','rejected','needs_investigation'):
        raise HTTPException(400,'Invalid evidence_status filter.')
    if source_stage not in (None,'evidence','medgemma'):
        raise HTTPException(400,'Invalid source_stage filter.')
    try:
        rows=(core.hybrid_search(data.query,document_id,evidence_status,source_stage,data.limit) if data.mode=='hybrid'
              else core.semantic_search(data.query,document_id,evidence_status,source_stage,data.limit) if data.mode=='semantic'
              else lexical_search(data.query,document_id,evidence_status,source_stage,data.limit))
    except ValueError as exc:
        raise HTTPException(409,str(exc))
    return {'query':data.query,'mode':data.mode,'filters':filters,'results':rows}

@app.post('/api/tools/get_source')
def tool_get_source(data: SourceRequest):
    try:
        return {'document_id':data.document_id,'pages':core.get_source(data.document_id,data.pages)}
    except ValueError as exc:
        raise HTTPException(404 if str(exc)=='Document not found.' else 400,str(exc))

@app.get('/api/tools/document_pages/{document_id}')
def tool_document_pages(document_id: str):
    try:
        return core.get_document_page_map(document_id)
    except ValueError as exc:
        raise HTTPException(404,str(exc))

@app.post('/api/tools/get_timeline')
def tool_get_timeline(data: TimelineRequest):
    if not 1<=data.limit<=1000:
        raise HTTPException(400,'limit must be between 1 and 1000.')
    if data.start and data.end and data.start>data.end:
        raise HTTPException(400,'start must not be after end.')
    return {'start':data.start,'end':data.end,'events':core.get_timeline(data.start,data.end,data.document_id,data.limit)}

@app.post('/api/tools/get_assessment_criteria')
def tool_get_assessment_criteria(data: CriteriaRequest):
    try:
        return core.get_assessment_criteria(data.topic)
    except ValueError as exc:
        raise HTTPException(400,str(exc))

def lexical_search(q, document_id=None, evidence_status=None, source_stage=None, limit=100):
    return core.lexical_search(q,document_id,evidence_status,source_stage,limit)
