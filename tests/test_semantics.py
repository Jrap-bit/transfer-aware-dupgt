"""Check semantic gradients, source-only retrieval, and saved model reuse."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_llm_transfer import fixture
from dupgt_cdr.common import write_json,file_sha256
from dupgt_cdr.teacher import write_jsonl,read_jsonl,seal_files,canonical
from dupgt_cdr.semantics import prepare_semantics,SemanticStore,SemanticDUPGTModel,adjust_gate
from dupgt_cdr.retrieval import (prepare_profiles,generate_profiles,build_retrieval,
                                 load_retrieval_pairs,validate_response,text_profile,
                                 build_text_teacher,load_text_only_pairs)
from dupgt_cdr.model import DUPGTModel
from dupgt_cdr.runtime import model_spec,save_checkpoint,load_saved_model,set_seed
from dupgt_cdr.predict import query_input,predict
from dupgt_cdr.train import parameter_hash
from dupgt_cdr.experiments import transfer_train,compare_experiments,lock_selection,transfer_evaluate


class TinyEncoder:
    # Only deterministic unit-test fixtures; no claim about semantic quality.
    info={"model":"unit-test-encoder","revision":"test","code_revision":"test",
          "max_length":128,"pooling":"unit-test","dimension":3}

    def encode(self,texts):
        values=np.array([[1+text.lower().count('space'),1+text.lower().count('romance'),
                          1+text.lower().count('family')] for text in texts],dtype=np.float32)
        return values/np.linalg.norm(values,axis=1,keepdims=True)


def metadata(folder,count):
    for domain in ('B','M'):
        write_jsonl(folder/f'{domain}.jsonl',[dict(asin=f'{domain}{i}',title=f'{domain} story {i}',
            description=('space '*((i%3)+1)+'romance '*((i%4)+1)+'family '*((i%5)+1)),
            categories='Fiction') for i in range(count)])


def cache_fixture(folder,count=12,source_only=3,**kwargs):
    data,manifest,config=fixture(item_count=count,source_only_count=source_only)
    config.update(max_shared_users=0,meta_epochs=1)
    metadata(folder,count)
    output=folder/'semantics'
    with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)):
        prepare_semantics(config,folder/'B.jsonl',folder/'M.jsonl',output,
                          encoder=TinyEncoder(),batch_size=5,max_length=128,**kwargs)
    return data,manifest,config,output


def teacher_fixture(folder,data,manifest,config,cache):
    output=folder/'profiles'
    with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)):
        prepare_profiles(config,cache,output,teacher_users=3)
        requests=read_jsonl(output/'requests.jsonl')
        external=folder/'external.jsonl'
        write_jsonl(external,[dict(request_id=r['request_id'],interests=[
            ['space space space adventures','romance romance family stories','family family family drama'][i%3]])
            for i,r in enumerate(requests)])
        def answer(base,route,payload):
            key=payload['response_format']['json_schema']['schema']['properties']['request_id']['const']
            response=next(r for r in read_jsonl(external) if r['request_id']==key)
            return {'model':'test-local','choices':[{'finish_reason':'stop','message':{'content':canonical(response)}}]}
        with patch('dupgt_cdr.retrieval.load_local_teacher',return_value={'model_digest':'test-weights'}), \
             patch('dupgt_cdr.retrieval.local_json',side_effect=answer):
            generate_profiles(output,model='test-local',runtime_file=folder/'runtime.json',
                              max_requests=10,max_total_tokens=100000)
        build_retrieval(output,top_k=3,encoder=TinyEncoder())
    return output


class SemanticTests(unittest.TestCase):
    def test_automatic_study_runs_both_models_and_controls_then_resumes_verified_stages(self):
        from dupgt_cdr.experiments import run_semantic_study
        from dupgt_cdr.settings import get_config
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            factor=folder/'factor.pt'
            factor.write_bytes(b'unit-test-factor')
            runtime=folder/'runtime.json'
            write_json(runtime,{'model_identifier':'test-local','model_key':'test-key'})
            config=get_config('2023','full')
            config['device']='cpu'
            semantic,teacher,output=folder/'semantic',folder/'teacher',folder/'output'
            calls=[]
            def invoke(arguments,**kwargs):
                if arguments[0]=='test-lms':
                    return
                calls.append(arguments[3])
                def value(flag):
                    return Path(arguments[arguments.index(flag)+1])
                command=arguments[3]
                if command=='semantic-prepare':
                    write_json(value('--output-dir')/'manifest.json',{'status':'completed'})
                elif command=='profile-prepare':
                    write_json(value('--output-dir')/'manifest.json',{'fixture':'prepared'})
                elif command=='profile-generate':
                    write_json(value('--teacher-dir')/'generation_summary.json',{'complete':True})
                elif command=='profile-retrieve':
                    write_json(value('--teacher-dir')/'controls.json',{'status':'ready'})
                elif command=='transfer-train':
                    self.assertEqual(str(value('--overlap-fraction')),'1.0')
                    write_json(value('--output-dir')/'artifact_manifest.json',{'fixture':'trained'})
                elif command=='transfer-compare':
                    write_json(value('--output-dir')/'comparison.json',{'fixture':'comparison'})
                elif command=='transfer-lock':
                    write_json(value('--selection-record'),{'fixture':'locked'})
                elif command=='transfer-evaluate':
                    write_json(value('--run-dir')/'test_metrics.json',{'fixture':'evaluated'})
                else:
                    raise AssertionError(command)
            with patch('shutil.which',return_value='test-lms'),patch('subprocess.run',side_effect=invoke), \
                 patch('dupgt_cdr.experiments.validated_run'):
                result=run_semantic_study(config,semantic,teacher,output,factor_checkpoint=factor,
                    runtime_file=runtime,model='test-local',encoding_device='cpu')
                self.assertEqual(result['status'],'completed')
                self.assertEqual(calls.count('transfer-train'),5)
                saved=list(calls)
                run_semantic_study(config,semantic,teacher,output,factor_checkpoint=factor,
                    runtime_file=runtime,model='test-local',encoding_device='cpu')
                self.assertEqual(calls,saved)
                with self.assertRaisesRegex(ValueError,'contract changed'):
                    run_semantic_study(config,semantic,teacher,output,factor_checkpoint=factor,
                        runtime_file=runtime,model='test-local',encoding_device='cpu',pseudo_weight=.1)

    def test_gate_equation_bounds_identity_and_finite_difference_gradients(self):
        base=torch.tensor([.1,.4,.8],dtype=torch.double,requires_grad=True)
        delta=torch.tensor([-2.,.3,1.],dtype=torch.double,requires_grad=True)
        valid=torch.ones(3,dtype=torch.bool)
        actual=adjust_gate(base,delta,valid)
        torch.testing.assert_close(actual,torch.sigmoid(torch.logit(base)+delta))
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:adjust_gate(a,b,valid),(base,delta)))
        torch.testing.assert_close(adjust_gate(base,torch.zeros_like(delta),valid),base,rtol=0,atol=0)
        extremes=adjust_gate(torch.tensor([0.,1.,.5,.5]),torch.tensor([100.,-100.,100.,-100.]),torch.ones(4,dtype=torch.bool))
        self.assertTrue(torch.isfinite(extremes).all())
        self.assertTrue(((extremes>=0)&(extremes<=1)).all())
        torch.testing.assert_close(adjust_gate(base,delta,torch.zeros(3,dtype=torch.bool)),base,rtol=0,atol=0)

    def test_pairwise_loss_sign_and_independent_mean_normalization(self):
        from dupgt_cdr.train import pairwise_loss
        a=torch.tensor([1.,2.],dtype=torch.double,requires_grad=True)
        b=torch.tensor([1.,0.],dtype=torch.double,requires_grad=True)
        loss=pairwise_loss(a,b)
        expected=(np.log(2)+np.log1p(np.exp(-2)))/2
        self.assertAlmostEqual(loss.item(),expected,places=12)
        loss.backward()
        self.assertTrue((a.grad<0).all())
        self.assertTrue((b.grad>0).all())
        torch.testing.assert_close(pairwise_loss(a.repeat(3),b.repeat(3)),loss)

    def test_text_control_preserves_rating_direction_and_neutrality(self):
        class BasisEncoder:
            info={'dimension':2}
            def encode(self,texts):
                return np.array([[1.,0.],[0.,1.],[1.,1.]])[:len(texts)]
        books=[{'rating':5,'title':'A'},{'rating':1,'title':'B'},{'rating':3,'title':'C'}]
        actual=text_profile(books,BasisEncoder())
        np.testing.assert_allclose(actual,np.array([1.,-1.])/np.sqrt(2))

    def test_encoding_resumes_and_checks_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            data,manifest,config,cache=cache_fixture(folder,max_batches=1)
            self.assertEqual(json.loads((cache/'manifest.json').read_text())['encoded_items'],5)
            with self.assertRaisesRegex(ValueError,'Finish semantic encoding'):
                SemanticStore(cache)
            with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)):
                prepare_semantics(config,folder/'B.jsonl',folder/'M.jsonl',cache,
                                  batch_size=5,max_length=128,encoder=TinyEncoder())
            store=SemanticStore(cache,manifest['benchmark_sha256'])
            self.assertEqual(len(store.ids),25)
            self.assertTrue(np.all(store.lookup([0])==0))
            np.testing.assert_allclose(np.linalg.norm(store.lookup(store.ids[1:]),axis=1),1,atol=1e-6)
            with self.assertRaisesRegex(ValueError,'another benchmark'):
                SemanticStore(cache,'other')
            changed=TinyEncoder()
            changed.info={**changed.info,'revision':'changed'}
            # Changed source inputs are rejected even after completion.
            (folder/'B.jsonl').write_text('{}\n')
            with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)):
                with self.assertRaisesRegex(ValueError,'inputs changed'):
                    prepare_semantics(config,folder/'B.jsonl',folder/'M.jsonl',cache,max_length=128,encoder=changed)

    def test_semantic_gate_starts_as_baseline_and_receives_rating_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,manifest,config,cache=cache_fixture(Path(tmp))
            spec=model_spec(data,config)
            set_seed(2020,1)
            baseline=DUPGTModel(**spec).eval()
            set_seed(2020,1)
            semantic=SemanticDUPGTModel(**spec,semantic_store=SemanticStore(cache)).eval()
            x=query_input(torch.as_tensor(data['validation'][:4]),torch.as_tensor(data['histories']),torch.device('cpu'))
            torch.testing.assert_close(baseline(x,'test_meta'),semantic(x,'test_meta'),rtol=0,atol=0)
            semantic.zero_grad()
            semantic(x,'test_meta').sum().backward()
            self.assertGreater(semantic.dupgt.semantic.network[-1].weight.grad.abs().sum().item(),0)
            # No source meaning or no candidate meaning must recover original DUPGT.
            with torch.no_grad():
                semantic.dupgt.semantic.network[-1].bias.fill_(2)
            x[:,2:]=0
            torch.testing.assert_close(baseline(x,'test_meta'),semantic(x,'test_meta'),rtol=0,atol=0)

    def test_profiles_keep_polarities_and_retrieval_only_returns_warm_movies(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,manifest,config,cache=cache_fixture(Path(tmp))
            store=SemanticStore(cache)
            profiles,count,total,coherence=store.profiles(data['histories'][:3])
            self.assertEqual(profiles.shape,(3,2,3))
            self.assertTrue(np.all(count<=total))
            ids,scores=store.retrieve(np.array([1,0,0]),3)
            self.assertTrue(set(ids)<=set(data['target_train'][:,1]))
            self.assertTrue(np.all(scores[:-1]>=scores[1:]))
            with self.assertRaises(ValueError):
                store.retrieve(np.zeros(3),3)

    def test_source_only_requests_and_fixture_teacher_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            data,manifest,config,cache=cache_fixture(folder)
            teacher=folder/'teacher'
            with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)):
                prepare_profiles(config,cache,teacher,teacher_users=3)
                requests=read_jsonl(teacher/'requests.jsonl')
                forbidden=set(data['shared_train_users'])|set(data['validation_users'])|set(data['test_users'])
                self.assertFalse(forbidden & {r['mapping']['user_id'] for r in requests})
                self.assertTrue(all(set(r['input'])=={'request_id','profile_token','books'} for r in requests))
                responses=folder/'external.jsonl'
                write_jsonl(responses,[dict(request_id=r['request_id'],interests=['A space exploration story.']) for r in requests])
                generate_profiles(teacher,model='fixture',model_version='test',responses_path=responses,
                                  fixture=True,max_requests=10,max_total_tokens=100000)
                with self.assertRaisesRegex(ValueError,'fixture labels'):
                    build_retrieval(teacher,top_k=3,encoder=TinyEncoder())

    def test_retrieval_builds_equal_budgets_and_validates_sealed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            data,manifest,config,cache=cache_fixture(folder)
            teacher=teacher_fixture(folder,data,manifest,config,cache)
            store=SemanticStore(cache)
            for kind in ('retrieval_llm','retrieval_text','retrieval_shuffled'):
                pairs,info=load_retrieval_pairs(teacher,kind,data,manifest,store.reference)
                self.assertEqual(pairs.shape,(9,3))
                self.assertTrue(np.all(pairs[:,1]!=pairs[:,2]))
            with self.assertRaisesRegex(ValueError,'different caches'):
                load_retrieval_pairs(teacher,'retrieval_llm',data,manifest,{})
            with (teacher/'pairs_retrieval_llm.jsonl').open('a') as handle:
                handle.write('{}\n')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_retrieval_pairs(teacher,'retrieval_llm',data,manifest,store.reference)

    def test_text_only_teacher_extends_llm_users_without_llm_and_trains(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            data,manifest,config,cache=cache_fixture(folder,source_only=6)
            llm=teacher_fixture(folder,data,manifest,config,cache)
            text=folder/'text_only'
            with patch('dupgt_cdr.data.get_benchmark',return_value=(data,manifest)):
                result=build_text_teacher(config,cache,text,teacher_users=6,top_k=3,encoder=TinyEncoder())
            store=SemanticStore(cache)
            small,large=read_jsonl(llm/'requests.jsonl'),read_jsonl(text/'requests.jsonl')
            self.assertGreater(len(large),len(small))
            self.assertEqual(large[:len(small)],small)  # the LLM users are a prefix of the larger set
            self.assertNotIn('responses.jsonl',result['files'])
            text_pairs={(r['user_id'],r['pair_id']):r for r in read_jsonl(llm/'pairs_retrieval_text.jsonl')}
            for row in read_jsonl(text/'pairs_retrieval_text_only.jsonl'):
                if (row['user_id'],row['pair_id']) in text_pairs:
                    old=text_pairs[(row['user_id'],row['pair_id'])]
                    self.assertEqual((row['preferred_movie_id'],row['other_movie_id']),
                                     (old['preferred_movie_id'],old['other_movie_id']))
            for kind in ('retrieval_text_only','retrieval_text_only_shuffled'):
                pairs,info=load_text_only_pairs(text,kind,data,manifest,store.reference)
                self.assertEqual(pairs.shape,(result['matched_pairs'],3))
                self.assertFalse(info['generation']['llm_called'])
            with self.assertRaisesRegex(ValueError,'text-only teacher kind'):
                load_retrieval_pairs(text,'retrieval_text',data,manifest,store.reference)
            with self.assertRaisesRegex(ValueError,'Not a text-only'):
                load_text_only_pairs(llm,'retrieval_text_only',data,manifest,store.reference)
            factor=folder/'factors.pt'
            save_checkpoint(factor,DUPGTModel(**model_spec(data,config)),model_spec(data,config),config,
                            manifest['benchmark_sha256'],'pretrained',1)
            with patch('dupgt_cdr.experiments.get_benchmark',return_value=(data,manifest)):
                transfer_train(config,str(factor),str(folder/'run'),overlap_fraction=.5,semantic_dir=cache,
                               teacher_kind='retrieval_text_only',teacher_dir=text,pseudo_weight=.1)
            experiment=json.loads((folder/'run'/'experiment.json').read_text())
            self.assertEqual(experiment['pseudo_pairs'],result['matched_pairs'])

    def test_semantic_training_checkpoint_reload_and_locked_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            data,manifest,config,cache=cache_fixture(folder)
            factor=folder/'factors.pt'
            save_checkpoint(factor,DUPGTModel(**model_spec(data,config)),model_spec(data,config),config,
                            manifest['benchmark_sha256'],'pretrained',1)
            teacher=teacher_fixture(folder,data,manifest,config,cache)
            paths=[folder/'baseline',folder/'semantics_only',folder/'semantics_llm']
            with patch('dupgt_cdr.experiments.get_benchmark',return_value=(data,manifest)):
                transfer_train(config,str(factor),str(paths[0]),overlap_fraction=.5)
                transfer_train(config,str(factor),str(paths[1]),overlap_fraction=.5,semantic_dir=cache)
                transfer_train(config,str(factor),str(paths[2]),overlap_fraction=.5,semantic_dir=cache,
                               teacher_kind='retrieval_llm',teacher_dir=teacher,pseudo_weight=.01)
                comparison=compare_experiments([str(p) for p in paths],str(folder/'comparison'))
                self.assertEqual([r['model_variant'] for r in comparison['rows']],['dupgt','semantics','semantics'])
                self.assertFalse(any((p/'test_metrics.json').exists() for p in paths))
                lock=folder/'selection.json'
                lock_selection([str(paths[1])],str(lock),'unit-test validation selection')
                transfer_evaluate(str(paths[1]),str(lock))
            checkpoint=torch.load(paths[2]/'checkpoints/dupgt_best.pt',weights_only=True)
            model=load_saved_model(checkpoint).eval()
            values,_=predict(model,data['validation'],torch.as_tensor(data['histories']),'test_meta',torch.device('cpu'),64)
            self.assertTrue(np.isfinite(values).all())
            with (paths[2]/'training_history.csv').open() as handle:
                history=list(csv.DictReader(handle))
            self.assertGreater(int(history[0]['pseudo_comparisons']),0)
            self.assertTrue(json.loads((paths[2]/'validation_metrics.json').read_text())['factors_unchanged'])
            write_json(cache/'manifest.json',{**json.loads((cache/'manifest.json').read_text()),'changed':True})
            with self.assertRaisesRegex(ValueError,'cache changed'):
                load_saved_model(checkpoint)

    def test_response_schema_refuses_ratings_foreign_ids_and_empty_descriptions(self):
        request={'request_id':'known'}
        self.assertEqual(validate_response({'request_id':'known','interests':[]},request),[])
        for response in ({'request_id':'other','interests':[]},
                         {'request_id':'known','interests':['short']},
                         {'request_id':'known','interests':[],'rating':5}):
            with self.assertRaises(ValueError):
                validate_response(response,request)


if __name__=='__main__':
    unittest.main()
