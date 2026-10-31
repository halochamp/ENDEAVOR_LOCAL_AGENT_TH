"""LLM-free Translate lifecycle regressions against the actual FastAPI routes."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient

import agent_server as srv


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'source.txt'
        self.source.write_text('hello')
        self.checkpoint = Path(self.tmp.name) / 'checkpoint.json'
        srv._translate_state = dict(job_id='test', status='paused', source_name='source.txt',
            source_path=str(self.source), checkpoint_path=str(self.checkpoint))
        srv._translate_control = dict(pause=threading.Event(), cancel=threading.Event())
        srv._translate_task = None

    def test_resume_through_fastapi_dispatch(self):
        async def worker(*args):
            srv._update_translate_state('test', status='done')
        # TestClient is used outside a "with" block, so the server's startup lifespan
        # (model server reconcile, history DB, watchdogs) never runs; auth is overridden.
        srv.api.dependency_overrides[srv._require_token] = lambda: None
        self.addCleanup(srv.api.dependency_overrides.pop, srv._require_token, None)
        with patch.object(srv, '_run_translate_job', worker):
            response = TestClient(srv.api).post('/translate/resume', params={'job_id': 'test'})
            self.assertEqual(response.status_code, 200)

    def test_resume_recovers_legacy_checkpoint_after_server_restart(self):
        job_id = 'translate-1791612544316-9ea286b6'
        private = Path(self.tmp.name) / 'private'
        uploads = private / 'uploads'
        uploads.mkdir(parents=True)
        source = uploads / 'th-translate-test.txt'
        source.write_text('hello')
        checkpoint = private / f'{job_id}.json'
        checkpoint.write_text(json.dumps({
            'source_name': 'source.txt',
            'source_sha256': srv.translate_documents._sha256(source),
            'model': 'current/no-think-model',
            'page_basis': 'logical_text',
            'next_index': 1,
            'translated': {'p1-b1': 'สวัสดี'},
        }))
        srv._translate_state = None
        srv._translate_control = None
        srv._translate_task = None
        observed = []

        async def worker(restored_job_id, restored_source, source_name, restored_checkpoint):
            observed.append((restored_job_id, restored_source, source_name, restored_checkpoint))
            srv._update_translate_state(restored_job_id, status='done', event='done')

        async def resume_and_wait():
            response = await srv.translate_resume(job_id)
            if srv._translate_task:
                await srv._translate_task
            return response

        with patch.object(srv, '_TRANSLATE_UPLOAD_DIR', uploads), \
                patch.object(srv._config, 'TRANSLATE_PRIVATE_DIR', str(private)), \
                patch.object(srv, '_run_translate_job', worker):
            response = asyncio.run(resume_and_wait())
            self.assertTrue(response['ok'])
            self.assertEqual(srv._translate_state['status'], 'done')
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0][0], job_id)
            self.assertEqual(observed[0][1], source)
            self.assertEqual(observed[0][2], 'source.txt')
            self.assertEqual(observed[0][3], checkpoint)
            self.assertTrue((private / 'active_job.json').exists())

    def test_cancel_done_is_idempotent(self):
        srv._translate_state['status'] = 'done'
        self.assertEqual(asyncio.run(srv.translate_cancel('test'))['job']['status'], 'done')

    def test_progress_preserves_pending_pause(self):
        srv._translate_state['status'] = 'pausing'
        srv._translate_progress('test', 'translate', 1, 1, {})
        self.assertEqual(srv._translate_state['status'], 'pausing')

    def test_failed_worker_keeps_resume_inputs(self):
        self.checkpoint.write_text('{}')
        with patch.object(srv.translate_documents, 'translate_document', side_effect=ValueError('bad batch')):
            asyncio.run(srv._run_translate_job('test', self.source, 'source.txt', self.checkpoint))
        self.assertEqual(srv._translate_state['status'], 'error')
        self.assertTrue(self.source.exists())
        self.assertTrue(self.checkpoint.exists())

    def test_cancel_queued_waits_for_worker_and_skips_extraction(self):
        srv._translate_state['status'] = 'queued'
        self.assertEqual(asyncio.run(srv.translate_cancel('test'))['job']['status'], 'cancelling')
        self.assertTrue(self.source.exists())
        with patch.object(srv.translate_documents, 'translate_document') as translate:
            asyncio.run(srv._run_translate_job('test', self.source, 'source.txt', self.checkpoint))
            translate.assert_not_called()
        self.assertEqual(srv._translate_state['status'], 'cancelled')
        self.assertFalse(self.source.exists())

    def test_cancel_during_upload_never_launches_worker(self):
        srv._translate_state = None
        class Request:
            headers = {'x-file-name': 'upload.txt'}
            async def stream(self):
                yield b'hello'
                await srv.translate_cancel()
        async def run():
            with patch.object(srv, '_TRANSLATE_UPLOAD_DIR', Path(self.tmp.name) / 'uploads'), \
                    patch.object(srv._config, 'TRANSLATE_PRIVATE_DIR', self.tmp.name), \
                    patch.object(srv, '_run_translate_job') as worker:
                response = await srv.translate_start(Request())
                self.assertEqual(response['job']['status'], 'cancelled')
                worker.assert_not_called()
                self.assertFalse(list((Path(self.tmp.name) / 'uploads').iterdir()))
        asyncio.run(run())

    def test_unknown_job_does_not_change_existing_job(self):
        response = asyncio.run(srv.translate_cancel('wrong'))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(srv._translate_state['status'], 'paused')


if __name__ == '__main__':
    unittest.main()
