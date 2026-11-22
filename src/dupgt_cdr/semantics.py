"""Cache item meanings and add semantic evidence to DUPGT's gate."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import resolve, write_json, file_sha256
from .model import DUPGTModel
from .teacher import check_files, identity, whitelist_metadata, write_jsonl

ENCODER = "jinaai/jina-embeddings-v2-base-en"
ENCODER_REVISION = "322d4d7e2f35e84137961a65af894fda0385eb7a"
# The encoder implementation lives in a separate upstream repository.
CODE_REVISION = "f3ec4cf7de7e561007f27c9efc7148b0bd713f81"
FEATURE_NAMES = ("similarity_high","similarity_low","count_high","count_low",
                 "coverage_high","coverage_low","coherence_high","coherence_low",
                 "has_high","has_low","has_movie_text","similarity_difference")


# 1. FROZEN TEXT ENCODER; BOTH DOMAINS USE THE SAME SPACE

class TextEncoder:
    def __init__(self, device="cpu", max_length=256, revision=ENCODER_REVISION,
                 code_revision=None):
        from .runtime import choose_device
        if not 8 <= max_length <= 8192:
            raise ValueError("Text length must be between 8 and 8192 tokens")
        cache = resolve("data/models/huggingface")
        torch.set_num_threads(4)
        os.environ.setdefault("HF_MODULES_CACHE", str(cache/"modules"))
        from transformers import AutoModel, AutoTokenizer
        # Pin both weights and executable model code, including on later resumes.
        code_revision = code_revision or CODE_REVISION
        self.info = {"model": ENCODER, "revision": revision, "code_revision": code_revision,
                     "max_length": max_length, "pooling": "masked_mean_l2", "dimension": 768}
        self.device, self.max_length = choose_device(device), max_length
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=revision, cache_dir=cache)
        self.model = AutoModel.from_pretrained(ENCODER, revision=revision,
            code_revision=code_revision, trust_remote_code=True, cache_dir=cache,
            use_safetensors=True).to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt").to(self.device)
        hidden = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        means = (hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        values = F.normalize(means, dim=1).cpu().float().numpy()
        if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) < .99):
            raise ValueError("Encoder produced missing or non-finite vectors")
        return values


def item_text(row):
    row = whitelist_metadata(row)
    return ". ".join(row[key] for key in ("title", "description", "categories") if row[key])


# 2. RESUMABLE ITEM CACHE; MISSING TEXT AND PADDING STAY ZERO

def prepare_semantics(config, book_metadata, movie_metadata, output_dir, *, device="cpu",
                      batch_size=16, max_length=256, limit_per_domain=0, max_batches=0,
                      encoder=None):
    from .data import get_benchmark
    if batch_size < 1 or limit_per_domain < 0 or max_batches < 0:
        raise ValueError("Invalid encoding batch or item limit")
    data, benchmark = get_benchmark(config)
    output = resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved_path = output/"manifest.json"
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else None
    if not saved and any(output.iterdir()):
        raise FileExistsError("Existing semantic directory has no resumable manifest")
    sources = {"book": str(resolve(book_metadata)), "movie": str(resolve(movie_metadata))}
    inputs = {"dataset_sha256": benchmark["benchmark_sha256"],
              "metadata_sha256": {k: file_sha256(Path(v)) for k,v in sources.items()},
              "metadata_paths": sources, "limit_per_domain": limit_per_domain,
              "text_version": "whitelisted-title-description-categories-v1"}
    if saved and any(saved[k] != v for k,v in inputs.items()):
        raise ValueError("Semantic cache inputs changed; use another directory")
    if saved and (saved["encoder"]["max_length"] != max_length or
                  (encoder is not None and saved["encoder"] != encoder.info)):
        raise ValueError("Semantic encoder changed on resume")
    if saved and saved["status"] == "completed":
        SemanticStore(output, benchmark["benchmark_sha256"])
        return saved
    encoder = encoder or TextEncoder(device, max_length,
        code_revision=saved["encoder"]["code_revision"] if saved else None)
    if saved and saved["encoder"] != encoder.info:
        raise ValueError("Semantic encoder changed on resume")
    dimension = encoder.info["dimension"]
    if not saved:
        n_source = len(data["source_item_ids"])
        allowed = {"book": {str(v):i+1 for i,v in enumerate(data["source_item_ids"])},
                   "movie": {str(v):i+n_source+1 for i,v in enumerate(data["target_item_ids"])}}
        warm = set(data["target_train"][:,1].tolist())
        rows, counts = [], {}
        for domain, source in sources.items():
            seen = set()
            with Path(source).open() as handle:
                for line in handle:
                    raw = json.loads(line)
                    if raw.get("dataset_release",config["dataset_release"]) != config["dataset_release"]:
                        raise ValueError("Metadata release differs from the rating benchmark")
                    row = whitelist_metadata(raw)
                    item = allowed[domain].get(str(row["asin"]))
                    if not item or (domain == "movie" and item not in warm):
                        continue
                    if item in seen:
                        raise ValueError("Duplicate item in cleaned metadata")
                    if not row["description"] or not row["title"]:
                        continue
                    seen.add(item)
                    rows.append({"item_id": item, "domain": domain, **row})
                    if limit_per_domain and len(seen) == limit_per_domain:
                        break
            counts[domain] = len(seen)
        if min(counts.values()) == 0:
            raise ValueError("Need descriptive metadata in both domains")
        rows.sort(key=lambda row: row["item_id"])
        write_jsonl(output/"items.jsonl", rows)
        ids = np.array([0]+[row["item_id"] for row in rows], dtype=np.int64)
        index = np.zeros(n_source+len(data["target_item_ids"])+1, dtype=np.int32)
        index[ids[1:]] = np.arange(1,len(ids))
        np.save(output/"item_ids.npy", ids, allow_pickle=False)
        np.save(output/"item_index.npy", index, allow_pickle=False)
        vectors = np.lib.format.open_memmap(output/"vectors.npy", mode="w+", dtype=np.float16,
                                           shape=(len(ids), dimension))
        vectors[:] = 0
        vectors.flush()
        saved = {**inputs, "schema":1, "status":"encoding", "config":config,
                 "encoder":encoder.info, "counts":counts, "encoded_items":0,
                 "n_source":n_source, "n_items":len(index)-1,
                 "catalogue_scope":"bounded_trial" if limit_per_domain else "all_clean_warm_metadata",
                 "input_files":{name:file_sha256(output/name) for name in
                                ("items.jsonl","item_ids.npy","item_index.npy")}}
        write_json(saved_path,saved)
    check_files(output, saved["input_files"])
    vectors = np.load(output/"vectors.npy", mmap_mode="r+", allow_pickle=False)
    start, batch, batches = saved["encoded_items"], [], 0
    def encode_batch():
        nonlocal start, batches
        values = encoder.encode([item_text(row) for row in batch])
        if values.shape != (len(batch),dimension) or not np.isfinite(values).all():
            raise ValueError("Bad encoder output shape or values")
        if np.any(np.abs(np.linalg.norm(values,axis=1)-1) > .01):
            raise ValueError("Text vectors must be nonzero and L2 normalized")
        vectors[start+1:start+1+len(batch)] = values
        vectors.flush()
        start += len(batch)
        batches += 1
        saved["encoded_items"] = start
        write_json(saved_path,saved)
        print(f"Text embeddings: {start:,}/{len(vectors)-1:,}",flush=True)
    with (output/"items.jsonl").open() as handle:
        for i,line in enumerate(handle):
            if i < start:
                continue
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                encode_batch()
                batch.clear()
                if max_batches and batches >= max_batches:
                    break
        else:
            if batch:
                encode_batch()
    if start == len(vectors)-1:
        saved["status"] = "completed"
        saved["files"] = {**saved["input_files"], "vectors.npy":file_sha256(output/"vectors.npy")}
        write_json(saved_path,saved)
    return saved


# 3. RETRIEVAL AND SOURCE-ONLY POLARITY PROFILES

class SemanticStore:
    def __init__(self, directory, dataset_sha256=None):
        self.directory = resolve(directory)
        self.manifest = json.loads((self.directory/"manifest.json").read_text())
        if self.manifest["status"] != "completed":
            raise ValueError("Finish semantic encoding before training or retrieval")
        if dataset_sha256 and self.manifest["dataset_sha256"] != dataset_sha256:
            raise ValueError("Semantic cache belongs to another benchmark")
        check_files(self.directory,self.manifest["files"])
        self.vectors = np.load(self.directory/"vectors.npy", mmap_mode="r", allow_pickle=False)
        self.index = np.load(self.directory/"item_index.npy", allow_pickle=False)
        self.ids = np.load(self.directory/"item_ids.npy", allow_pickle=False)
        if (self.vectors.shape != (len(self.ids),self.manifest["encoder"]["dimension"]) or
            self.index.shape != (self.manifest["n_items"]+1,) or
            np.any(self.index[self.ids] != np.arange(len(self.ids))) or
            np.any(self.vectors[0] != 0)):
            raise ValueError("Semantic IDs, padding or vector dimensions differ")
        self.reference = {"directory":str(self.directory),
                          "manifest_sha256":file_sha256(self.directory/"manifest.json")}

    def lookup(self, ids):
        ids = np.asarray(ids,dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= len(self.index)):
            raise ValueError("Unknown semantic item ID")
        values = np.asarray(self.vectors[self.index[ids]],dtype=np.float32)
        # Float16 storage slightly changes lengths; restore true cosine geometry.
        return values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-8)

    def retrieve(self, query, top_k=10):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = np.asarray(query,dtype=np.float32)
        norm = np.linalg.norm(query)
        if query.shape != (self.vectors.shape[1],) or not np.isfinite(query).all() or norm < 1e-8:
            raise ValueError("Retrieval needs a nonempty query in the encoder space")
        ids = self.ids[self.ids > self.manifest["n_source"]]
        scores = self.lookup(ids) @ (query/norm)
        # Item ID resolves exact ties reproducibly.
        order = np.lexsort((ids,-scores))[:top_k]
        return ids[order],scores[order]

    def profiles(self, histories):
        history = np.asarray(histories,dtype=np.int64)
        if history.ndim != 2 or history.shape[1] != 40 or np.any(history < 0) or np.any(history > self.manifest["n_source"]):
            raise ValueError("Semantic profiles require the 40 source Book history slots")
        vectors = self.lookup(history).reshape(len(history),2,20,-1)
        covered = np.linalg.norm(vectors,axis=-1) > 0
        count = covered.sum(2)
        means = vectors.sum(2)/np.maximum(count[...,None],1)
        coherence = np.linalg.norm(means,axis=-1)
        profiles = means/np.maximum(coherence[...,None],1e-8)
        total = (history.reshape(-1,2,20) != 0).sum(2)
        return profiles.astype(np.float32), count, total, coherence

    def features(self, x):
        values = x.detach().cpu().numpy()
        profiles,count,total,coherence = self.profiles(values[:,2:42])
        movie = self.lookup(values[:,1])
        present = np.linalg.norm(movie,axis=-1) > 0
        similarities = (profiles*movie[:,None,:]).sum(-1)
        stats = np.column_stack((similarities,np.log1p(count)/np.log(21),
            count/np.maximum(total,1),coherence,count>0,present,similarities[:,0]-similarities[:,1]))
        valid = present & (count.sum(1)>0)
        return profiles,movie,stats.astype(np.float32),valid


# 4. LEARNED SEMANTIC GATE; RATINGS REMAIN THE MAIN OBJECTIVE

class SemanticGate(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.projection = nn.Linear(dimension,16,bias=False)
        self.network = nn.Sequential(nn.Linear(60,32),nn.ELU(),nn.Linear(32,1))
        # Begin at exactly the original DUPGT gate.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, profiles,movie,stats):
        projected = self.projection(torch.cat((profiles,movie[:,None,:]),dim=1)).flatten(1)
        return self.network(torch.cat((projected,stats),dim=1)).squeeze(1)


def adjust_gate(base,delta,valid):
    # Equivalent to sigmoid(logit(base) + delta), including endpoint gates.
    odds = torch.exp(delta.clamp(-10,10))
    updated = base*odds/(1-base+base*odds).clamp_min(1e-12)
    return torch.where(valid,updated,base)


class SemanticDUPGTModel(DUPGTModel):
    def __init__(self, n_users,n_items,embedding_dim, semantic_store):
        super().__init__(n_users,n_items,embedding_dim)
        self.semantic_store = semantic_store
        self.dupgt.semantic = SemanticGate(semantic_store.vectors.shape[1])
        self.model_variant = "semantics"

    def forward_with_details(self,x):
        source = self.src_model.uid_embedding(x[:,0]).unsqueeze(1)
        target = self.tgt_model.iid_embedding(x[:,1])
        sequence = x[:,2:42]
        features = self.src_model.iid_embedding(sequence)
        high = self.dupgt.bridging_high(features[:,:20],sequence[:,:20])
        low = self.dupgt.bridging_low(features[:,20:],sequence[:,20:])
        base = self.dupgt.gating_mlp(high,low,source.squeeze(1),target)
        profiles,movie,stats,valid = self.semantic_store.features(x)
        tensors = [torch.as_tensor(a,device=x.device,dtype=target.dtype) for a in (profiles,movie,stats)]
        delta = self.dupgt.semantic(*tensors)
        alpha = adjust_gate(base,delta,torch.as_tensor(valid,device=x.device))
        bridge = (alpha[:,None]*high+(1-alpha[:,None])*low).reshape(-1,self.emb_dim,self.emb_dim)
        user = torch.bmm(source,bridge).squeeze(1)
        rating = (user*target).sum(1)
        return rating,alpha,{"base_alpha_high":base,"semantic_log_odds":delta,
                             "semantic_features":tensors[2],"has_semantics":torch.as_tensor(valid,device=x.device)}

    def forward_with_gate(self,x):
        rating,alpha,_ = self.forward_with_details(x)
        return rating,alpha
