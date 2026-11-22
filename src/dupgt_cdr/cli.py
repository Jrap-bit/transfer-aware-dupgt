"""Prepare data, train DUPGT, or run an offline teacher experiment."""

from __future__ import annotations

import argparse
import json

from .settings import get_config, TRANSFER
from .common import resolve


# Route a command to data, training, or prediction code.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "pipeline", "recommend", "verify", "compare",
                        "teacher-prepare", "teacher-generate", "teacher-controls", "transfer-train",
                        "transfer-lock", "transfer-evaluate", "transfer-compare",
                        "teacher-audit-prepare", "teacher-audit-generate", "teacher-audit-compare", "metadata-prepare", "transfer-grid",
                        "semantic-prepare","profile-prepare","profile-generate","profile-retrieve","text-teacher-build","semantic-run",
                        "uncertainty-fit","uncertainty-evaluate","probabilistic-train"])
    parser.add_argument("--dataset", choices=["2014", "2023"], default="2014")
    parser.add_argument("--size", choices=["smoke", "progress", "full"], default="full")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                        help="Training device; overrides settings.py")
    parser.add_argument("--output-dir", help="New training output directory; existing runs are never overwritten")
    parser.add_argument("--user-id", help="Raw Amazon reviewer ID; default is the first cold test user")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--runs", nargs="+", help="Completed run directories for compare")
    # Offline teacher settings.
    parser.add_argument("--book-metadata", help="Release-matched metadata path or HTTPS URL")
    parser.add_argument("--movie-metadata", help="Release-matched metadata path or HTTPS URL")
    parser.add_argument("--doh", action="store_true", help="Use DNS over HTTPS for metadata downloads if system DNS fails")
    parser.add_argument("--minimum-books", type=int, default=3, help="Clean Books needed for the metadata coverage count")
    parser.add_argument("--teacher-dir")
    parser.add_argument("--teacher-users", type=int, default=TRANSFER["teacher_users"])
    parser.add_argument("--teacher-seed", type=int, default=TRANSFER["teacher_seed"])
    parser.add_argument("--llm-model", help="Loaded LM Studio identifier; e.g. dupgt-qwen35-9b")
    parser.add_argument("--server-url", default="http://127.0.0.1:1234")
    parser.add_argument("--runtime-file", default=TRANSFER["runtime_file"], help="Verified local model setup record")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=TRANSFER["max_output_tokens"])
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--context-tokens", type=int, default=TRANSFER["context_tokens"])
    parser.add_argument("--dry-run", action="store_true")
    # One shared semantic cache for both new model variants.
    parser.add_argument("--semantic-dir",help="Completed item embedding cache")
    parser.add_argument("--encoding-device",choices=["cpu","mps","cuda"],default="mps")
    parser.add_argument("--encoding-batch-size",type=int,default=16)
    parser.add_argument("--text-length",type=int,default=256)
    parser.add_argument("--limit-per-domain",type=int,default=0,help="Bounded encoder trial; zero encodes all clean items")
    parser.add_argument("--max-batches",type=int,default=0,help="Pause encoding after this many batches; zero finishes it")
    parser.add_argument("--responses", help="Import structured JSONL responses instead of calling LM Studio")
    parser.add_argument("--model-version", help="Model digest/version for imported responses")
    parser.add_argument("--fixture", action="store_true", help="Mark imported mock labels; disallow experiment training")
    parser.add_argument("--audit-dir", help="Separate validation-only teacher audit directory")
    parser.add_argument("--audit-users", type=int, default=8)
    parser.add_argument("--models", nargs="+", help="Local model identifiers to compare in the audit")
    # Bridge experiments use separate seeds and explicit final selection.
    parser.add_argument("--factor-checkpoint")
    parser.add_argument("--overlap-fraction", type=float, default=0.01)
    parser.add_argument("--overlap-seed", type=int, default=TRANSFER["overlap_seed"])
    parser.add_argument("--transfer-seed", type=int, default=TRANSFER["transfer_seed"])
    parser.add_argument("--teacher-kind", choices=["none", "text", "llm", "shuffled","retrieval_llm","retrieval_text","retrieval_shuffled",
                        "retrieval_text_only","retrieval_text_only_shuffled"], default="none")
    parser.add_argument("--pseudo-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-batch-size", type=int, default=TRANSFER["pseudo_batch_size"])
    parser.add_argument("--epochs", type=int, help="Override bridge epochs for a bounded trial")
    parser.add_argument("--gate-prior-weight", type=float, default=0.01,
                        help="probabilistic-train only: Gaussian prior on the gate log-odds (stops gate collapse)")
    parser.add_argument("--run-dir")
    parser.add_argument("--selection-record")
    parser.add_argument("--reason", help="Validation-based reason for locking chosen runs")
    parser.add_argument("--final", action="store_true", help="Compare explicitly evaluated final test results")
    parser.add_argument("--across-overlap", action="store_true")
    args = parser.parse_args()
    def require(*names):
        for name in names:
            if not getattr(args, name):
                parser.error(f"{args.command} requires --{name.replace('_', '-')}")

    if args.command == "teacher-audit-generate":
        from .teacher_audit import generate_audit
        require("audit_dir")
        if args.max_requests < 1:
            parser.error("teacher-audit-generate requires a positive --max-requests")
        generate_audit(resolve(args.audit_dir), resolve(args.runtime_file), args.max_requests,
                       args.context_tokens, args.max_output_tokens)
        return
    if args.command == "teacher-audit-compare":
        from .teacher_audit import compare_audits
        require("audit_dir", "models")
        compare_audits(resolve(args.audit_dir), args.models)
        return
    if args.command == "teacher-generate":
        from .teacher import generate_teacher
        require("teacher_dir")
        generate_teacher(resolve(args.teacher_dir), model=args.llm_model, base_url=args.server_url,
                         max_requests=args.max_requests, max_output_tokens=args.max_output_tokens,
                         max_total_tokens=args.max_total_tokens, context_tokens=args.context_tokens,
                         dry_run=args.dry_run, responses_path=resolve(args.responses) if args.responses else None,
                         model_version=args.model_version, fixture=args.fixture,
                         runtime_file=resolve(args.runtime_file))
        return
    if args.command == "profile-generate":
        from .retrieval import generate_profiles
        require("teacher_dir")
        result=generate_profiles(args.teacher_dir,model=args.llm_model,base_url=args.server_url,
            runtime_file=args.runtime_file,max_requests=args.max_requests,max_output_tokens=args.max_output_tokens,
            max_total_tokens=args.max_total_tokens,context_tokens=args.context_tokens,dry_run=args.dry_run,
            responses_path=args.responses,model_version=args.model_version,fixture=args.fixture)
        print(result)
        return
    if args.command == "profile-retrieve":
        from .retrieval import build_retrieval
        require("teacher_dir")
        result=build_retrieval(args.teacher_dir,device=args.device or "cpu",top_k=args.top_k,seed=args.teacher_seed)
        print({k:v for k,v in result.items() if k not in {"files","users"}})
        return
    if args.command == "teacher-controls":
        from .teacher import build_controls
        require("teacher_dir")
        build_controls(resolve(args.teacher_dir), args.teacher_seed)
        return
    if args.command == "transfer-lock":
        from .experiments import lock_selection
        require("runs", "selection_record", "reason")
        lock_selection(args.runs, args.selection_record, args.reason)
        return
    if args.command == "transfer-evaluate":
        from .experiments import transfer_evaluate
        require("run_dir", "selection_record")
        transfer_evaluate(args.run_dir, args.selection_record)
        return
    if args.command == "uncertainty-fit":
        from .uncertainty import fit_blend
        require("runs", "output_dir")
        result = fit_blend(args.runs, args.output_dir)
        print(json.dumps(result["validation"], indent=2))
        return
    if args.command == "uncertainty-evaluate":
        from .uncertainty import evaluate_blend
        require("run_dir", "selection_record")
        result = evaluate_blend(args.run_dir, args.selection_record)
        print(json.dumps({k: v for k, v in result.items() if k != "single_bridges"}, indent=2))
        return
    if args.command == "transfer-compare":
        from .experiments import compare_experiments
        require("runs", "output_dir")
        compare_experiments(args.runs, args.output_dir, final=args.final, across_overlap=args.across_overlap)
        return
    # Compare results from finished runs.
    if args.command == "compare":
        from .evaluate import compare
        if not args.runs:
            parser.error("compare requires --runs RUN1 RUN2")
        compare(args.runs, args.output_dir or "outputs/comparison/new_comparison")
        return
    config = get_config(args.dataset, args.size)
    if args.device:
        config["device"] = args.device
    if args.command == "semantic-run":
        from .experiments import run_semantic_study
        require("semantic_dir","teacher_dir","output_dir")
        config["device"]=args.device or "cpu"
        if args.epochs is not None:
            if args.epochs<1:
                parser.error("--epochs must be positive")
            config["meta_epochs"]=args.epochs
        factor=args.factor_checkpoint or f"outputs/amazon{args.dataset}/full_seed2020/checkpoints/pretrained_factors.pt"
        run_semantic_study(config,args.semantic_dir,args.teacher_dir,args.output_dir,
            factor_checkpoint=factor,pseudo_weight=args.pseudo_weight or .01,teacher_users=args.teacher_users,
            encoding_device=args.encoding_device,model=args.llm_model or TRANSFER["llm_model"],runtime_file=args.runtime_file)
        return
    if args.command == "semantic-prepare":
        from .semantics import prepare_semantics
        require("output_dir")
        metadata=f"data/metadata/amazon{args.dataset}/clean"
        result=prepare_semantics(config,args.book_metadata or f"{metadata}/books.jsonl",
            args.movie_metadata or f"{metadata}/movies.jsonl",args.output_dir,device=args.device or "cpu",
            batch_size=args.encoding_batch_size,max_length=args.text_length,
            limit_per_domain=args.limit_per_domain,max_batches=args.max_batches)
        print({k:v for k,v in result.items() if k not in {"config","files","input_files"}})
        return
    if args.command == "profile-prepare":
        from .retrieval import prepare_profiles
        require("semantic_dir","output_dir")
        result=prepare_profiles(config,args.semantic_dir,args.output_dir,args.teacher_users,
                                args.teacher_seed,args.minimum_books)
        print({k:v for k,v in result.items() if k not in {"config","files","prompt"}})
        return
    if args.command == "text-teacher-build":
        from .retrieval import build_text_teacher
        require("semantic_dir","output_dir")
        result=build_text_teacher(config,args.semantic_dir,args.output_dir,teacher_users=args.teacher_users,
            seed=args.teacher_seed,top_k=args.top_k,minimum_books=args.minimum_books,device=args.device or "cpu")
        print({k:v for k,v in result.items() if k not in {"files","users","semantic_cache"}})
        return
    if args.command == "metadata-prepare":
        from .metadata import prepare_metadata
        require("output_dir")
        prepare_metadata(config, args.output_dir, args.book_metadata, args.movie_metadata,
                         args.doh, args.minimum_books)
        return
    factor = args.factor_checkpoint or f"outputs/amazon{args.dataset}/full_seed2020/checkpoints/pretrained_factors.pt"
    if args.command == "teacher-audit-prepare":
        from .teacher_audit import prepare_audit
        require("book_metadata", "movie_metadata", "output_dir")
        prepare_audit(config, args.book_metadata, args.movie_metadata, args.output_dir,
                      args.audit_users, args.teacher_seed, args.minimum_books)
        return
    if args.command == "teacher-prepare":
        from .teacher import prepare_teacher
        require("book_metadata", "movie_metadata", "output_dir")
        prepare_teacher(config, args.book_metadata, args.movie_metadata, args.output_dir,
                        factor, args.teacher_users, args.teacher_seed, args.minimum_books)
        return
    if args.command == "probabilistic-train":
        from .probabilistic import probabilistic_train
        require("output_dir", "semantic_dir")
        if args.epochs is not None:
            if args.epochs < 1:
                parser.error("--epochs must be positive")
            config["meta_epochs"] = args.epochs
        probabilistic_train(config, factor, args.output_dir, semantic_dir=args.semantic_dir,
                            overlap_fraction=args.overlap_fraction, overlap_seed=args.overlap_seed,
                            transfer_seed=args.transfer_seed, gate_prior_weight=args.gate_prior_weight)
        return
    if args.command in {"transfer-train", "transfer-grid"}:
        from .experiments import transfer_train, run_transfer_grid
        require("output_dir")
        if args.epochs is not None:
            if args.epochs < 1:
                parser.error("--epochs must be positive")
            config["meta_epochs"] = args.epochs
        if args.command == "transfer-grid":
            if args.semantic_dir:
                parser.error("Use transfer-train for semantic variants; transfer-grid is the earlier pairwise pilot")
            require("teacher_dir")
            run_transfer_grid(config, factor, args.teacher_dir, args.output_dir,
                              args.overlap_seed, args.transfer_seed)
            return
        transfer_train(config, factor, args.output_dir, overlap_fraction=args.overlap_fraction,
                       overlap_seed=args.overlap_seed, transfer_seed=args.transfer_seed,
                       teacher_kind=args.teacher_kind, teacher_dir=args.teacher_dir,
                       pseudo_weight=args.pseudo_weight, pseudo_batch_size=args.pseudo_batch_size,
                       semantic_dir=args.semantic_dir)
        return
    if args.command in {"prepare", "pipeline"}:
        from .data import prepare
        prepare(config)
    if args.command in {"train", "pipeline"}:
        from .train import train
        train(config, args.output_dir)
    elif args.command == "recommend":
        from .predict import infer
        if args.top_k < 1:
            parser.error("--top-k must be positive")
        infer(config, args.output_dir, args.user_id, args.top_k)
    elif args.command == "verify":
        from .data import get_benchmark, assert_prepared_integrity
        arrays, manifest = get_benchmark(config)
        checks = assert_prepared_integrity(arrays)
        print(f"{manifest['dataset']}: {manifest['cohort']['shared_users']:,} shared users, "
              f"{manifest['splits']['test']['ratings']:,} test ratings")
        for name, passed in checks.items():
            print(f"  {name}: {passed}")


if __name__ == "__main__":
    main()

