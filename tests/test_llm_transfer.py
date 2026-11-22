"""Tiny fixtures for teacher integrity and bridge-only experiments; no API calls."""
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dupgt_cdr.common import file_sha256, write_json
from dupgt_cdr.data import build_benchmark
from dupgt_cdr.experiments import (array_hash, load_factor_initialized_model,
                                   select_overlap_rows, source_only_user_ids)
from dupgt_cdr.model import DUPGTModel
from dupgt_cdr.runtime import model_spec, save_checkpoint, set_seed
from dupgt_cdr.teacher import (PROMPT, PROMPT_VERSION, canonical, extract_metadata,
    generation_config, identity, item_asin, matched_controls, normalize_response,
    retained_books, text_teacher, whitelist_metadata, validate_teacher_artifacts, generate_teacher,
    write_jsonl, seal_files, load_local_teacher, preference_payload, local_url)
from dupgt_cdr.train import fit_transfer, pairwise_loss, parameter_hash
from dupgt_cdr.teacher_audit import prepare_audit, generate_audit, score_choices, compare_audits


def fixture(item_count=6, source_only_count=2):
    def domain(users, prefix):
        rows = [(u, i, 1+(u+i)%5) for u in range(len(users)) for i in range(item_count)]
        return {'users': np.array(users), 'items': np.array([f'{prefix}{i}' for i in range(item_count)]),
                'u': np.array([r[0] for r in rows]), 'i': np.array([r[1] for r in rows]),
                'r': np.array([r[2] for r in rows]), 't': np.arange(len(rows))}
    config = dict(seed=5, max_shared_users=20, max_source_only_users=-1, max_target_only_users=-1,
                  validation_fraction=.1, test_fraction=.2, label='fixture', dataset_release='2014',
                  embedding_dim=3, torch_threads=1, meta_epochs=2, mapping_epochs=1, meta_batch_size=16,
                  batch_size=16, evaluation_batch_size=64, meta_learning_rate=.01, mapping_learning_rate=.01,
                  weight_decay=0., device='cpu')
    users = [f'U{i:02d}' for i in range(20)]
    data, manifest = build_benchmark(domain(users+[f'source{i+1}' for i in range(source_only_count)], 'B'), domain(users+['target'], 'M'), config)
    manifest['benchmark_sha256'] = 'fixture-benchmark'
    return data, manifest, config


def request_fixture(user=0, token='u0'):
    payload = {'profile_token': token,
               'books': [{'alias': 'b0', 'rating': 5, 'title': 'Space adventure', 'description': 'aliens rocket science', 'categories': 'fiction'}],
               'movies': [{'alias': 'm0', 'title': 'Space adventure', 'description': 'aliens rocket science', 'categories': 'fiction'},
                          {'alias': 'm1', 'title': 'Romantic drama', 'description': 'wedding marriage romance', 'categories': 'drama'}],
               'pairs': [{'pair_id': 'p0', 'a': 'm0', 'b': 'm1'}]}
    key = identity({'input': payload, 'prompt': PROMPT, 'version': PROMPT_VERSION})
    return {'request_id': key, 'input': {'request_id': key, **payload},
            'mapping': {'user_id': user, 'movies': {'m0': 7, 'm1': 8}, 'books': {'b0': 1}}}


class AuditTests(unittest.TestCase):
    def test_grid_resumes_after_comparison_failure_without_retraining(self):
        from dupgt_cdr.experiments import run_transfer_grid, compare_experiments
        from dupgt_cdr.settings import TRANSFER
        data, manifest, config = fixture()
        config.update(dataset_release='2023', max_shared_users=0, meta_epochs=1)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            factor = folder/'factors.pt'
            model = DUPGTModel(**model_spec(data,config))
            save_checkpoint(factor,model,model_spec(data,config),config,manifest['benchmark_sha256'],'pretrained',1)
            teacher = folder/'teacher'
            teacher.mkdir()
            controls = {'status':'ready','personalization_identifiable':True}
            def controls_ready(*args):
                write_json(teacher/'controls.json',controls)
                return controls
            def tiny_pairs(*args):
                users = source_only_user_ids(data)
                pairs = np.array([[int(users[0]),7,8],[int(users[1]),8,7]],dtype=np.int64)
                return pairs, {'support_sha256':'same-pairs','controls_sha256':file_sha256(teacher/'controls.json'),
                    'coverage':{'requested_pairs':2,'matched_pairs':2,'changed_label_fraction':.5}}
            output = folder/'grid'
            with patch.dict(TRANSFER,overlap_fractions=[.5],pseudo_weights=[.01,.1]), \
                 patch('dupgt_cdr.teacher.build_controls',side_effect=controls_ready), \
                 patch('dupgt_cdr.teacher.load_training_pairs',side_effect=tiny_pairs), \
                 patch('dupgt_cdr.experiments.get_benchmark',return_value=(data,manifest)):
                with patch('dupgt_cdr.experiments.compare_experiments',side_effect=ValueError('aggregation failed')):
                    with self.assertRaisesRegex(ValueError,'aggregation failed'):
                        run_transfer_grid(config,str(factor),str(teacher),str(output))
                saved = {p:file_sha256(p) for p in output.glob('overlap_*/*/checkpoints/dupgt_best.pt')}
                self.assertEqual(len(saved),7)
                with patch('dupgt_cdr.experiments.transfer_train',side_effect=AssertionError('must reuse saved models')):
                    result = run_transfer_grid(config,str(factor),str(teacher),str(output))
                    repeated = run_transfer_grid(config,str(factor),str(teacher),str(output))
                self.assertEqual(result['status'],'validation_complete_final_selection_pending')
                self.assertEqual(len(result['reused_runs']),7)
                self.assertEqual(len(result['groups'][0]['validation_comparisons']),2)
                self.assertEqual(repeated['groups'],result['groups'])
                self.assertEqual(saved,{p:file_sha256(p) for p in saved})
                self.assertFalse(list(output.glob('overlap_*/*/test_metrics.json')))
                with self.assertRaisesRegex(ValueError,'pseudo weight'):
                    compare_experiments(result['groups'][0]['runs'],str(folder/'invalid-mixed-weights'))
                with self.assertRaisesRegex(ValueError,'training configuration differs'):
                    run_transfer_grid({**config,'meta_epochs':2},str(factor),str(teacher),str(output))

    def test_grid_refuses_empty_or_unpersonalized_teacher_support(self):
        from dupgt_cdr.experiments import run_transfer_grid
        with tempfile.TemporaryDirectory() as temporary:
            for i, controls in enumerate(({'status':'insufficient_matched_support','personalization_identifiable':False},
                                          {'status':'ready','personalization_identifiable':False})):
                output = Path(temporary)/f'grid{i}'
                with patch('dupgt_cdr.teacher.build_controls',return_value=controls), \
                     patch('dupgt_cdr.experiments.transfer_train') as train:
                    with self.assertRaises(ValueError):
                        run_transfer_grid({},'unused-factors',temporary,str(output))
                    train.assert_not_called()
                self.assertEqual(json.loads((output/'grid_summary.json').read_text())['status'],'blocked_teacher_support')

    def test_2023_teacher_preparation_and_fresh_bridge_use_matching_factors(self):
        from dupgt_cdr.teacher import prepare_teacher
        from dupgt_cdr.experiments import transfer_train
        data, manifest, config = fixture(item_count=12, source_only_count=3)
        config.update(dataset_release='2023', max_shared_users=0, meta_epochs=1)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for domain in ('B', 'M'):
                write_jsonl(folder/f'{domain}.jsonl', [dict(parent_asin=f'{domain}{i}', asin='wrong-child',
                    title=f'{domain} story {i}', description='A scientist travels to a distant planet and discovers a hidden civilization while searching for her family.',
                    categories=['Science fiction']) for i in range(12)])
            factor = folder/'factors.pt'
            model = DUPGTModel(**model_spec(data,config))
            save_checkpoint(factor,model,model_spec(data,config),config,manifest['benchmark_sha256'],'pretrained',1)
            teacher_dir = folder/'teacher'
            with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)):
                result = prepare_teacher(config,str(folder/'B.jsonl'),str(folder/'M.jsonl'),
                    str(teacher_dir),str(factor),teacher_users=2)
                self.assertEqual(result['status'],'ready')
                self.assertEqual(validate_teacher_artifacts(data,manifest,teacher_dir)['requests'],2)
            with patch('dupgt_cdr.experiments.get_benchmark',return_value=(data,manifest)):
                transfer_train(config,str(factor),str(folder/'run'),overlap_fraction=.1)
            self.assertTrue((folder/'run'/'validation_metrics.json').exists())
            self.assertFalse((folder/'run'/'test_metrics.json').exists())
            foreign = {**manifest, 'benchmark_sha256':'different-release-id-mapping'}
            with self.assertRaises(ValueError):
                load_factor_initialized_model(factor,data,foreign,config,2020)

    def test_audit_generation_resumes_without_repeating_a_successful_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            requests = [request_fixture(token=f'u{i}') for i in range(2)]
            for request in requests:
                request['mode'] = 'original'
            write_jsonl(directory/'requests.jsonl', requests)
            # Generation must not parse outcome values, even when the file is not JSON.
            (directory/'hidden_outcomes.jsonl').write_text('hidden outcomes are not prompt data')
            write_json(directory/'manifest.json', dict(purpose='validation_teacher_audit_only',
                never_training_labels=True, prompt=PROMPT, prompt_version=PROMPT_VERSION,
                files=seal_files(directory, ['requests.jsonl', 'hidden_outcomes.jsonl'])))
            runtime = directory/'runtime.json'
            write_json(runtime, dict(model_identifier='model', server_url='http://127.0.0.1:1234'))
            def answer(base, route, payload):
                key = payload['response_format']['json_schema']['schema']['properties']['request_id']['const']
                return dict(model='model', choices=[dict(finish_reason='stop', message=dict(content=canonical(dict(
                    request_id=key, preferences=[dict(pair_id='p0', choice='A')]))))])
            with patch('dupgt_cdr.teacher_audit.load_local_teacher', return_value={'model_digest':'weights'}), \
                 patch('dupgt_cdr.teacher_audit.local_json', side_effect=answer) as api:
                self.assertFalse(generate_audit(directory, runtime, max_requests=1)['complete'])
                self.assertTrue(generate_audit(directory, runtime, max_requests=1)['complete'])
                self.assertTrue(generate_audit(directory, runtime, max_requests=1)['complete'])
                self.assertEqual(api.call_count, 2)
                with patch('dupgt_cdr.teacher_audit.PROMPT', 'different task'):
                    with self.assertRaisesRegex(ValueError, 'prompt changed'):
                        generate_audit(directory, runtime, max_requests=1)

    def test_validation_inputs_keep_ratings_hidden_and_shuffle_profiles(self):
        data, manifest, config = fixture()
        config['max_shared_users'] = 0
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for domain in ('B', 'M'):
                write_jsonl(folder/f'{domain}.jsonl', [dict(asin=f'{domain}{i}',
                    title=f'{domain} title {i}', description='A descriptive catalogue story',
                    categories='Fiction') for i in range(6)])
            output = folder/'audit'
            with patch('dupgt_cdr.data.get_benchmark', return_value=(data, manifest)):
                result = prepare_audit(config, str(folder/'B.jsonl'), str(folder/'M.jsonl'), str(output), users=2)
            requests = [json.loads(line) for line in (output/'requests.jsonl').read_text().splitlines()]
            self.assertEqual(result['purpose'], 'validation_teacher_audit_only')
            self.assertTrue(result['never_training_labels'])
            with self.assertRaisesRegex(ValueError, 'cannot be used'):
                validate_teacher_artifacts(data, manifest, output)
            self.assertEqual(len(requests), 4)
            self.assertTrue(set(result['files']) >= {'hidden_outcomes.jsonl', 'requests.jsonl'})
            for original, shuffled in zip(requests[:2], requests[2:]):
                self.assertIn(original['mapping']['user_id'], data['validation_users'])
                self.assertNotIn(original['mapping']['user_id'], data['test_users'])
                self.assertEqual(original['input']['movies'], shuffled['input']['movies'])
                self.assertEqual(original['input']['pairs'], shuffled['input']['pairs'])
                self.assertNotEqual(original['profile_user_id'], shuffled['profile_user_id'])
                public = canonical(preference_payload(original, 'model', 512))
                self.assertNotIn('correct_choice', public)
                self.assertNotIn('rating_a', public)
                self.assertNotIn('user_id', public)
                self.assertTrue(all('rating' not in movie for movie in original['input']['movies']))

    def test_abstentions_and_invalid_answers_keep_the_full_denominator(self):
        request = request_fixture()
        request['mode'] = 'original'
        outcomes = [dict(profile_token='u0', pair_id='p0', correct_choice='A')]
        response = dict(request_id=request['request_id'], valid=True,
                        response=dict(request_id=request['request_id'], preferences=[dict(pair_id='p0', choice='insufficient_evidence')]))
        abstention = score_choices([request], [response], outcomes, 'original')
        self.assertEqual(abstention['pairs'], 1)
        self.assertEqual(abstention['coverage'], 0)
        self.assertIsNone(abstention['accuracy_when_answered'])
        response['response']['preferences'][0]['choice'] = 'A'
        self.assertEqual(score_choices([request], [response], outcomes, 'original')['correct_pairs'], 1)
        response['valid'] = False
        self.assertEqual(score_choices([request], [response], outcomes, 'original')['correct_fraction_of_all_pairs'], 0)

    def test_comparison_locks_answers_before_scoring_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            request = request_fixture()
            request['mode'] = 'original'
            write_jsonl(directory/'requests.jsonl', [request])
            write_jsonl(directory/'hidden_outcomes.jsonl', [dict(profile_token='u0', pair_id='p0',
                rating_a=5, rating_b=1, correct_choice='A', popularity_choice='B')])
            write_json(directory/'manifest.json', dict(users=1, purpose='validation_teacher_audit_only',
                never_training_labels=True, files=seal_files(directory, ['requests.jsonl', 'hidden_outcomes.jsonl'])))
            for model in ('one', 'two'):
                output = directory/model
                output.mkdir()
                write_json(output/'generation.json', dict(model=model))
                write_jsonl(output/'responses.jsonl', [dict(request_id=request['request_id'], mode='original',
                    valid=True, elapsed_seconds=1., response=dict(request_id=request['request_id'],
                    preferences=[dict(pair_id='p0', choice='A')]))])
                write_json(output/'summary.json', dict(complete=True,
                    files=seal_files(output, ['generation.json', 'responses.jsonl'])))
            # The retained single teacher can also be audited against cheap controls.
            import shutil
            single = directory/'single'
            single.mkdir()
            for name in ('manifest.json','requests.jsonl','hidden_outcomes.jsonl'):
                shutil.copyfile(directory/name,single/name)
            shutil.copytree(directory/'one',single/'one')
            single_report = compare_audits(single,['one'])
            self.assertEqual(single_report['results'][0]['accuracy_when_answered'],1.)
            report = compare_audits(directory, ['one', 'two'])
            self.assertTrue((directory/'prediction_lock.json').exists())
            self.assertEqual(report['results'][0]['accuracy_when_answered'], 1.)
            with (directory/'one'/'responses.jsonl').open('a') as handle:
                handle.write('{}\n')
            with self.assertRaises(ValueError):
                compare_audits(directory, ['one', 'two'])


class SelectionTests(unittest.TestCase):
    def test_nested_train_only_and_unchanged(self):
        data, _, _ = fixture()
        before = {k: array_hash(v) for k, v in data.items()}
        small, _ = select_overlap_rows(data, .1, 42)
        larger, rows = select_overlap_rows(data, .5, 42)
        self.assertTrue(set(small) <= set(larger) <= set(data['shared_train_users']))
        self.assertFalse(set(rows[:, 0]) & set(data['test_users']))
        self.assertEqual(before, {k: array_hash(v) for k, v in data.items()})
        self.assertEqual(len(source_only_user_ids(data)), 2)
        self.assertFalse(set(source_only_user_ids(data)) & set(data['shared_train_users']))

    def test_id_round_trip_and_retained_real_ratings(self):
        data, _, _ = fixture()
        self.assertEqual(item_asin(data, 1, 'book'), 'B0')
        self.assertEqual(item_asin(data, 7, 'movie'), 'M0')
        with self.assertRaises(ValueError):
            item_asin(data, 0, 'book')
        users = source_only_user_ids(data)
        profiles = retained_books(data, users)
        expected = {(int(u), int(i)): int(r) for u, i, r in data['source_train']}
        for user, parts in profiles.items():
            for item, rating in parts[0]+parts[1]:
                self.assertEqual(rating, expected[(user, item)])


class TeacherTests(unittest.TestCase):
    def test_whitelist_and_legacy_metadata(self):
        row = dict(asin='B0', title='Book', description='A story', categories=[['Books','Fiction']],
                   average_rating=5, rating_count=100, related={'also_bought':['X']}, reviewText='secret')
        self.assertEqual(set(whitelist_metadata(row)), {'asin','title','description','categories'})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'metadata.gz'
            with gzip.open(path, 'wt') as handle:
                handle.write(repr(row)+'\n')
                handle.write(json.dumps(row)+'\n')
                handle.write(json.dumps({**row, 'title':'Conflicting'})+'\n')
            lookup, audit = extract_metadata(str(path), {'B0'})
            self.assertFalse(lookup)
            self.assertEqual(audit['ambiguous_asins'], ['B0'])

    def test_strict_response_and_abstention(self):
        request = request_fixture()
        response = {'request_id': request['request_id'], 'preferences':[{'pair_id':'p0','choice':'A'}]}
        self.assertEqual(normalize_response(response, request)[0]['preferred_movie_id'], 7)
        for choice in ['tie', 'insufficient_evidence']:
            self.assertEqual(normalize_response({**response,'preferences':[{'pair_id':'p0','choice':choice}]}, request), [])
        for bad in [{**response,'request_id':'foreign'}, {**response,'preferences':[{'pair_id':'p9','choice':'A'}]},
                    {**response,'preferences':response['preferences']*2},
                    {**response,'preferences':[{'pair_id':'p0','choice':'C'}]}]:
            with self.assertRaises(ValueError):
                normalize_response(bad, request)
        broken = copy.deepcopy(request)
        broken['mapping']['movies']['m1'] = 7
        with self.assertRaises(ValueError):
            normalize_response(response, broken)

    def test_cache_identity_tracks_model_and_metadata(self):
        first = generation_config('qwen3:4b','digest1',512,8192)
        second = generation_config('qwen3:4b','digest2',512,8192)
        self.assertNotEqual(identity(first), identity(second))
        self.assertNotEqual(identity({'model':first,'metadata':'a'}), identity({'model':first,'metadata':'b'}))
        self.assertNotEqual(identity({'model':first,'prompt':'a'}), identity({'model':first,'prompt':'b'}))

    def test_text_and_matched_shuffle(self):
        requests = [request_fixture(i, f'u{i}') for i in range(20)]
        text = text_teacher(requests)
        self.assertEqual(len(text),20)
        self.assertTrue(all(r['preferred_movie_id']==7 for r in text))
        llm = []
        for i, request in enumerate(requests):
            llm += normalize_response({'request_id':request['request_id'],
                'preferences':[{'pair_id':'p0','choice':'A' if i%2 else 'B'}]}, request)
        a,b,c,fraction = matched_controls(llm,text,42)
        self.assertEqual(len(a),len(b))
        self.assertEqual(sorted(r['preferred_movie_id'] for r in a),sorted(r['preferred_movie_id'] for r in c))
        self.assertEqual(matched_controls(llm,text,42)[2],c)
        self.assertGreater(fraction,0)



def saved_teacher(directory, data, manifest, config):
    users = source_only_user_ids(data)
    metadata, requests = [], []
    for domain, ids, offset in [('book',data['source_item_ids'],0), ('movie',data['target_item_ids'],len(data['source_item_ids']))]:
        for i, asin in enumerate(ids):
            metadata.append({'domain':domain,'asin':str(asin),'title':str(asin),
                             'description':'space adventure' if i%2 else 'romantic drama','categories':'fiction'})
    lookup={(r['domain'],r['asin']):r for r in metadata}
    profiles=retained_books(data,users)
    for n,user in enumerate(users):
        item,rating=(profiles[int(user)][0]+profiles[int(user)][1])[0]
        book=lookup[('book',item_asin(data,item,'book'))]
        movies=[lookup[('movie',item_asin(data,i,'movie'))] for i in [7,8]]
        payload={'profile_token':f'u{n}',
                 'books':[{'alias':'b0','rating':rating,**{k:v for k,v in book.items() if k not in {'domain','asin'}}}],
                 'movies':[{'alias':f'm{i}',**{k:v for k,v in movie.items() if k not in {'domain','asin'}}} for i,movie in enumerate(movies)],
                 'pairs':[{'pair_id':'p0','a':'m0','b':'m1'}]}
        key=identity({'input':payload,'prompt':PROMPT,'version':PROMPT_VERSION})
        requests.append({'request_id':key,'input':{'request_id':key,**payload},
                         'mapping':{'user_id':int(user),'books':{'b0':item},'movies':{'m0':7,'m1':8}}})
    write_jsonl(directory/'requests.jsonl',requests)
    write_jsonl(directory/'metadata.jsonl',metadata)
    write_json(directory/'selection.json',{'selected_users':users.tolist(),'movie_pool':[7,8]})
    artifact={'status':'ready','dataset_sha256':manifest['benchmark_sha256'],'factor_sha256':'fixture',
              'config':config,'prompt':PROMPT,'prompt_version':PROMPT_VERSION,
              'files':seal_files(directory,['requests.jsonl','metadata.jsonl','selection.json'])}
    write_json(directory/'manifest.json',artifact)
    return requests


class ArtifactTests(unittest.TestCase):
    def test_prepare_import_controls_and_fixture_exclusion(self):
        from dupgt_cdr.teacher import prepare_teacher,build_controls,load_training_pairs
        data,manifest,config=fixture(item_count=12,source_only_count=20)
        config['max_shared_users']=0
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            factors=root/'factors.pt'
            model=DUPGTModel(**model_spec(data,config))
            save_checkpoint(factors,model,model_spec(data,config),config,manifest['benchmark_sha256'],'pretrained',1)
            paths=[]
            for domain,key in [('book','source_item_ids'),('movie','target_item_ids')]:
                path=root/f'{domain}.jsonl'
                write_jsonl(path,[{'asin':str(asin),'title':'Space adventure' if i%2 else 'Romantic drama',
                                  'description':'aliens rockets science' if i%2 else 'wedding marriage romance',
                                  'categories':['Fiction'],'average_rating':5,'review_count':999} for i,asin in enumerate(data[key])])
                paths.append(str(path))
            directory=root/'teacher'
            with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)):
                artifact=prepare_teacher(config,*paths,str(directory),str(factors),teacher_users=20)
                self.assertEqual(artifact['status'],'ready')
                requests=[json.loads(line) for line in (directory/'requests.jsonl').read_text().splitlines()]
                self.assertEqual(len(requests),20)
                self.assertEqual(len(requests[0]['input']['pairs']),5)
                self.assertNotIn('average_rating',(directory/'requests.jsonl').read_text())
                imported=root/'responses.jsonl'
                write_jsonl(imported,[{'request_id':r['request_id'],'preferences':[
                    {'pair_id':pair['pair_id'],'choice':'A' if i%2 else 'B'} for pair in r['input']['pairs']]}
                    for i,r in enumerate(requests)])
                generate_teacher(directory,model='fixture',model_version='fixture',responses_path=imported,
                                 max_requests=20,max_total_tokens=1000000,fixture=True)
                controls=build_controls(directory)
                self.assertGreaterEqual(controls['matched_pairs'],20)
                with self.assertRaisesRegex(ValueError,'Fixture'):
                    load_training_pairs(directory,'llm',data,manifest,artifact['factor_sha256'])

    def test_valid_requests_reject_checksum_and_heldout_user(self):
        data,manifest,config=fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            requests=saved_teacher(directory,data,manifest,config)
            self.assertEqual(validate_teacher_artifacts(data,manifest,directory)['requests'],2)
            requests[0]['mapping']['user_id']=int(data['test_users'][0])
            write_jsonl(directory/'requests.jsonl',requests)
            with self.assertRaises(ValueError):
                validate_teacher_artifacts(data,manifest,directory)
            artifact=json.loads((directory/'manifest.json').read_text())
            artifact['files']['requests.jsonl']=file_sha256(directory/'requests.jsonl')
            write_json(directory/'manifest.json',artifact)
            with self.assertRaises(ValueError):
                validate_teacher_artifacts(data,manifest,directory)

    def test_local_generation_resumes_without_repeating_calls(self):
        data,manifest,config=fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            saved_teacher(directory,data,manifest,config)
            calls=[]
            def local(base,route,payload=None):
                calls.append(payload)
                request_id=payload['response_format']['json_schema']['schema']['properties']['request_id']['const']
                return {'model':'local:4b','choices':[{'finish_reason':'stop','message':{
                    'content':json.dumps({'request_id':request_id,'preferences':[{'pair_id':'p0','choice':'A'}]})}}],
                    'usage':{'prompt_tokens':100,'completion_tokens':30}}
            runtime={'model_digest':'fixture-digest','runtime_manifest_sha256':'fixture-runtime',
                     'app_version':'fixture-version','loaded_context_tokens':8192}
            with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)), \
                 patch('dupgt_cdr.teacher.local_json',side_effect=local), \
                 patch('dupgt_cdr.teacher.load_local_teacher',return_value=runtime):
                first=generate_teacher(directory,model='local:4b',max_requests=2,max_total_tokens=100000,fixture=True,runtime_file=directory/'runtime.json')
                second=generate_teacher(directory,model='local:4b',max_requests=2,max_total_tokens=100000,fixture=True,runtime_file=directory/'runtime.json')
            self.assertEqual(len(calls),2)
            self.assertEqual(second['cache_hits'],2)
            self.assertEqual(first['accepted_pairs'],2)
            self.assertEqual(first['reserved_tokens'],second['reserved_tokens'])

    def test_wrong_response_model_cannot_create_teacher_labels(self):
        data,manifest,config=fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            saved_teacher(directory,data,manifest,config)
            runtime={'model_digest':'fixture-digest','runtime_manifest_sha256':'fixture-runtime'}
            wrong={'model':'foreign-model','choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
            with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)), \
                 patch('dupgt_cdr.teacher.load_local_teacher',return_value=runtime), \
                 patch('dupgt_cdr.teacher.local_json',return_value=wrong):
                summary=generate_teacher(directory,model='local-model',max_requests=2,max_total_tokens=100000,
                                         runtime_file=directory/'runtime.json',fixture=True)
            self.assertEqual(summary['accepted_pairs'],0)
            self.assertIn('Wrong response model',(directory/'rejections.jsonl').read_text())

    def test_conflicting_imports_and_abstentions_never_become_labels(self):
        data,manifest,config=fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            requests=saved_teacher(directory,data,manifest,config)
            response={'request_id':requests[0]['request_id'],'preferences':[{'pair_id':'p0','choice':'A'}]}
            rows=[response,{**response,'preferences':[{'pair_id':'p0','choice':'B'}]},
                  {'request_id':requests[1]['request_id'],'preferences':[{'pair_id':'p0','choice':'tie'}]}]
            path=directory/'import.jsonl'
            write_jsonl(path,rows)
            with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)):
                summary=generate_teacher(directory,model='external',model_version='fixture-version',
                    responses_path=path,max_requests=10,max_total_tokens=100000,fixture=True)
            self.assertEqual(summary['accepted_pairs'],0)
            self.assertEqual(summary['status'],'failed_empty_teacher')
            self.assertIn('duplicate',(directory/'rejections.jsonl').read_text())


class LocalRuntimeTests(unittest.TestCase):
    def test_weights_model_and_context_match_verified_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            weights=directory/'model.gguf'
            weights.write_bytes(b'fixture-weights')
            runtime={'runner':'LM Studio','status':'ready','structured_output_verified':True,
                     'model_identifier':'local-model','model_key':'test-key','server_url':'http://127.0.0.1:1234',
                     'model_path':str(weights),'sha256':file_sha256(weights),'app_version':'fixture'}
            path=directory/'runtime.json'
            write_json(path,runtime)
            models={'models':[{'key':'test-key','format':'gguf','size_bytes':weights.stat().st_size,
                               'loaded_instances':[{'id':'local-model','config':{'context_length':8192}}]}]}
            with patch('dupgt_cdr.teacher.local_json',return_value=models):
                details=load_local_teacher(path,'local-model','http://127.0.0.1:1234',8192)
                self.assertEqual(details['model_digest'],runtime['sha256'])
                with self.assertRaisesRegex(ValueError,'context'):
                    load_local_teacher(path,'local-model','http://127.0.0.1:1234',32768)
                with self.assertRaises(ValueError):
                    load_local_teacher(path,'other-model','http://127.0.0.1:1234',8192)
                weights.write_bytes(b'changed-weights')
                with self.assertRaisesRegex(ValueError,'weights changed'):
                    load_local_teacher(path,'local-model','http://127.0.0.1:1234',8192)

    def test_payload_keeps_private_mapping_local(self):
        request=request_fixture(user=77)
        payload=preference_payload(request,'local-model',512)
        self.assertEqual(payload['reasoning_effort'],'none')
        self.assertEqual(payload['response_format']['json_schema']['schema']['properties']['request_id']['const'],request['request_id'])
        self.assertNotIn('mapping',canonical(payload))
        self.assertNotIn('user_id',canonical(payload))
        self.assertEqual(payload['max_tokens'],512)
        self.assertEqual([m['role'] for m in payload['messages']],['system','user'])

    def test_remote_endpoints_are_rejected(self):
        for address in ['https://api.example.com','http://192.168.1.1:1234','http://user:password@localhost:1234']:
            with self.assertRaises(ValueError):
                local_url(address)


class ExperimentFlowTests(unittest.TestCase):
    def test_validation_then_lock_and_explicit_evaluation(self):
        from dupgt_cdr.experiments import transfer_train,transfer_evaluate,lock_selection,compare_experiments
        data,manifest,config=fixture()
        config['max_shared_users']=0
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            factors=directory/'factors.pt'
            model=DUPGTModel(**model_spec(data,config))
            save_checkpoint(factors,model,model_spec(data,config),config,manifest['benchmark_sha256'],'pretrained',1)
            runs=[]
            with patch('dupgt_cdr.experiments.get_benchmark',return_value=(data,manifest)):
                for fraction in [.1,.5]:
                    run=directory/f'run{fraction}'
                    transfer_train(config,str(factors),str(run),overlap_fraction=fraction)
                    self.assertFalse((run/'test_metrics.json').exists())
                    self.assertFalse((run/'test_predictions.csv.gz').exists())
                    runs.append(str(run))
                with self.assertRaises(FileNotFoundError):
                    transfer_evaluate(runs[0],str(directory/'missing.json'))
                with self.assertRaises(ValueError):
                    compare_experiments(runs,str(directory/'wrong'))
                compared=compare_experiments(runs,str(directory/'comparison'),across_overlap=True)
                self.assertEqual(compared['axis'],'overlap_fraction')
                selection=directory/'locked.json'
                lock_selection(runs,str(selection),'Synthetic workflow check; not research evidence')
                result=transfer_evaluate(runs[0],str(selection))
                self.assertIn('raw',result)
                repeated=transfer_evaluate(runs[0],str(selection))
                self.assertEqual(result,repeated)
                with self.assertRaises(ValueError):
                    lock_selection(runs,str(directory/'late.json'),'Too late')


class TrainingTests(unittest.TestCase):
    def test_pair_loss_direction_and_extreme_margin(self):
        preferred = torch.tensor([0.],requires_grad=True)
        loss = pairwise_loss(preferred,torch.tensor([0.]))
        loss.backward()
        self.assertLess(float(preferred.grad),0)
        self.assertLess(float(pairwise_loss(torch.tensor([5.]),torch.tensor([0.]))),float(loss.detach()))
        self.assertTrue(torch.isfinite(pairwise_loss(torch.tensor([-1e6]),torch.tensor([1e6]))))

    def test_pair_gradient_improves_an_actual_model_margin(self):
        from dupgt_cdr.predict import query_input
        data,_,config=fixture()
        set_seed(22,1)
        model=DUPGTModel(**model_spec(data,config)).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.dupgt.parameters():
            parameter.requires_grad_(True)
        user=int(source_only_user_ids(data)[0])
        history=torch.as_tensor(data['histories'])
        a=query_input(torch.tensor([[user,7]]),history,torch.device('cpu'))
        b=query_input(torch.tensor([[user,8]]),history,torch.device('cpu'))
        optimizer=torch.optim.SGD(model.dupgt.parameters(),lr=.001)
        before=float((model(a,'test_meta')-model(b,'test_meta')).detach())
        for _ in range(5):
            optimizer.zero_grad()
            pairwise_loss(model(a,'test_meta'),model(b,'test_meta')).backward()
            optimizer.step()
        after=float((model(a,'test_meta')-model(b,'test_meta')).detach())
        self.assertGreater(after,before)

    def test_fresh_bridge_and_factor_mismatch(self):
        data, manifest, config = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'factors.pt'
            set_seed(10,1)
            original = DUPGTModel(**model_spec(data,config))
            save_checkpoint(path,original,model_spec(data,config),config,manifest['benchmark_sha256'],'pretrained',1)
            a = load_factor_initialized_model(path,data,manifest,config,42)
            b = load_factor_initialized_model(path,data,manifest,config,42)
            self.assertEqual(parameter_hash(a,('dupgt.',)),parameter_hash(b,('dupgt.',)))
            self.assertEqual(parameter_hash(a,('src_model.','tgt_model.')),parameter_hash(original,('src_model.','tgt_model.')))
            self.assertNotEqual(parameter_hash(a,('dupgt.',)),parameter_hash(original,('dupgt.',)))
            with self.assertRaises(ValueError):
                load_factor_initialized_model(path,data,{'benchmark_sha256':'wrong'},config,42)

    def test_zero_weight_compatibility_frozen_factors_and_no_test_prediction(self):
        from dupgt_cdr.predict import predict as real_predict
        data, manifest, config = fixture()
        states = []
        pairs = np.array([[source_only_user_ids(data)[0],7,8]])
        with tempfile.TemporaryDirectory() as temporary:
            for run, pseudo in enumerate([None,pairs]):
                set_seed(17,1)
                model = DUPGTModel(**model_spec(data,config))
                initial = parameter_hash(model,('dupgt.',))
                factor_before = parameter_hash(model,('src_model.','tgt_model.','aug_model.'))
                output = Path(temporary)/str(run)
                output.mkdir()
                log = []
                def validation_only(model,rows,*args,**kwargs):
                    self.assertIs(rows,data['validation'])
                    return real_predict(model,rows,*args,**kwargs)
                with patch('dupgt_cdr.train.predict',side_effect=validation_only):
                    fit_transfer(model,data,config,torch.device('cpu'),'dupgt',log,output,
                        model_spec(data,config),manifest['benchmark_sha256'],pseudo_pairs=pseudo,pseudo_weight=0)
                self.assertNotEqual(initial,parameter_hash(model,('dupgt.',)))
                self.assertEqual(factor_before,parameter_hash(model,('src_model.','tgt_model.','aug_model.')))
                states.append({k:v.clone() for k,v in model.state_dict().items()})
            for key in states[0]:
                torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)
            with self.assertRaises(ValueError):
                fit_transfer(model,data,config,torch.device('cpu'),'dupgt',[],output,
                    model_spec(data,config),manifest['benchmark_sha256'],pseudo_pairs=[],pseudo_weight=.1)

    def test_auxiliary_training_updates_only_bridge(self):
        data, manifest, config = fixture()
        set_seed(19,1)
        model = DUPGTModel(**model_spec(data,config))
        before = parameter_hash(model,('src_model.','tgt_model.','aug_model.'))
        pairs = np.array([[source_only_user_ids(data)[0],7,8]])
        with tempfile.TemporaryDirectory() as temporary:
            log=[]
            fit_transfer(model,data,config,torch.device('cpu'),'dupgt',log,Path(temporary),
                         model_spec(data,config),manifest['benchmark_sha256'],real_rows=data['meta_train'][:8],
                         pseudo_pairs=pairs,pseudo_weight=.1,transfer_seed=19,pseudo_batch_size=4)
            self.assertEqual(before,parameter_hash(model,('src_model.','tgt_model.','aug_model.')))
            self.assertTrue(all(r['pair_loss']>0 and r['pseudo_comparisons']==4 for r in log))


if __name__ == '__main__':
    unittest.main()
