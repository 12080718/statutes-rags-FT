#!/usr/bin/env python3
"""
4択法令データ（デジタル庁）を用いたRAG評価スクリプト
"""
import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import re
import unicodedata
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# プロジェクトルートの.envファイルを読み込み
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from app.core.rag_config import load_config
from app.core.prompts import build_mc_prompt_cot, build_mc_prompt_direct
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.rag_pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker


def create_retriever(config):
    """設定に基づいてRetrieverを作成"""
    retriever_type = config.retriever.retriever_type
    index_path = Path(config.vector_store_path)
    
    if retriever_type == "vector":
        return VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda,
            mmr_fetch_k_max=config.retriever.mmr_fetch_k_max
        )
    elif retriever_type == "bm25":
        return BM25Retriever(
            index_path=str(index_path / "bm25"),
            tokenizer=config.retriever.bm25_tokenizer
        )
    else:
        vector_retriever = VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda,
            mmr_fetch_k_max=config.retriever.mmr_fetch_k_max
        )
        bm25_retriever = BM25Retriever(
            index_path=str(index_path / "bm25"),
            tokenizer=config.retriever.bm25_tokenizer
        )
        return HybridRetriever(
            vector_retriever, 
            bm25_retriever,
            fusion_method=config.retriever.fusion_method,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight,
            rrf_k=config.retriever.rrf_k,
            fetch_k_multiplier=config.retriever.fetch_k_multiplier
        )


def _parse_choices(raw: Any) -> Dict[str, str]:
    """
    lawqa形式の選択肢（文字列/辞書）を a/b/c/d の辞書に揃える。
    """
    if isinstance(raw, dict):
        return {k.lower(): str(v) for k, v in raw.items() if k.lower() in {"a", "b", "c", "d"}}
    choices: Dict[str, str] = {}
    if isinstance(raw, str):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            key = line[:1].lower()
            if key in {"a", "b", "c", "d"}:
                text = line[1:].lstrip(").．.、　 ").strip()
                choices[key] = text
    for k in ["a", "b", "c", "d"]:
        choices.setdefault(k, "")
    return choices


def create_cot_prompt(question: str, choices: Any, context: str = "") -> str:
    """
    CoT用プロンプト生成（prompts.py の共通関数で日本語指示を生成）。
    """
    choices_dict = _parse_choices(choices)
    return build_mc_prompt_cot(question, choices_dict, context, style="compact")


def create_multiple_choice_prompt(question: str, choices: Any, context: str = "", use_few_shot: bool = True) -> str:
    """
    4択直接回答プロンプト生成（prompts.py の共通関数で日本語指示を生成）。
    """
    choices_dict = _parse_choices(choices)
    return build_mc_prompt_direct(question, choices_dict, context, few_shot=use_few_shot)


CHOICE_MAP = {"1": "a", "2": "b", "3": "c", "4": "d"}
VALID_CHOICES = {"a", "b", "c", "d"}


def _extract_choice_from_text(text: str) -> Optional[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    m = re.search(r"(?<![0-9])([1-4])(?![0-9])", normalized)
    if m:
        return CHOICE_MAP[m.group(1)]
    m = re.search(r"(?<![a-z])([abcd])(?![a-z])", normalized)
    if m:
        return m.group(1)
    return None


def normalize_and_parse_answer(raw_output: str) -> Optional[str]:
    """
    LLM 出力から a/b/c/d を抽出する（見つからなければ None）。
    - Answer/回答 行を末尾から優先
    - 次に末尾数行、最後に全文でフォールバック
    """
    if not raw_output:
        return None

    text = unicodedata.normalize("NFKC", raw_output).lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for line in reversed(lines):
        if "answer" in line or "回答" in line:
            choice = _extract_choice_from_text(line)
            if choice in VALID_CHOICES:
                return choice

    tail = "\n".join(lines[-3:]) if len(lines) >= 3 else "\n".join(lines)
    choice = _extract_choice_from_text(tail)
    if choice in VALID_CHOICES:
        return choice

    choice = _extract_choice_from_text(text)
    if choice in VALID_CHOICES:
        return choice

    return None


def extract_answer(response: str, is_cot: bool = False) -> str:
    """LLM応答から回答(a/b/c/d)を抽出"""
    choice = normalize_and_parse_answer(response)
    return choice if choice is not None else "unknown"


def evaluate_sample(pipeline: RAGPipeline, sample: Dict[str, Any], use_rag: bool = True, use_few_shot: bool = True, use_cot: bool = False) -> Dict[str, Any]:
    """1サンプルを評価"""
    question = sample['問題文']
    choices = sample['選択肢']
    correct_answer = sample['output'].lower()
    
    if use_rag:
        # RAGで関連法令を検索
        documents = pipeline.retrieve_documents(question)
        context = pipeline.format_context(documents)
    else:
        # コンテキストなし（LLMのみ）
        documents = []
        context = "法令条文が提供されていません。あなたの知識に基づいて回答してください。"
    
    # プロンプトを作成
    if use_cot:
        prompt = create_cot_prompt(question, choices, context)
    else:
        prompt = create_multiple_choice_prompt(question, choices, context, use_few_shot=use_few_shot)
    
    # LLMに質問
    try:
        response = pipeline.llm.invoke(prompt)
        predicted_answer = extract_answer(response, is_cot=use_cot)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        predicted_answer = "error"
        response = str(e)
    
    # 正解判定
    is_correct = predicted_answer == correct_answer
    
    result = {
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "response": response,
        "retrieved_docs_count": len(documents),
        "file_name": sample.get('ファイル名', ''),
        "references": sample.get('references', [])
    }
    
    # CoT使用時は推論プロセスも記録
    if use_cot:
        result["reasoning"] = response
    
    return result


def evaluate_sample_with_ensemble(pipeline: RAGPipeline, sample: Dict[str, Any], 
                                 n_runs: int = 3, use_rag: bool = True, 
                                 use_few_shot: bool = True, use_cot: bool = False) -> Dict[str, Any]:
    """アンサンブル評価（複数回推論して多数決）"""
    from collections import Counter
    
    question = sample['問題文']
    choices = sample['選択肢']
    correct_answer = sample['output'].lower()
    
    # RAG検索は1回のみ
    if use_rag:
        documents = pipeline.retrieve_documents(question)
        context = pipeline.format_context(documents)
    else:
        documents = []
        context = "法令条文が提供されていません。あなたの知識に基づいて回答してください。"
    
    # 複数回推論
    predictions = []
    responses = []
    
    for i in range(n_runs):
        # プロンプト作成
        if use_cot:
            prompt = create_cot_prompt(question, choices, context)
        else:
            prompt = create_multiple_choice_prompt(question, choices, context, use_few_shot=use_few_shot)
        
        try:
            response = pipeline.llm.invoke(prompt)
            predicted = extract_answer(response, is_cot=use_cot)
            predictions.append(predicted)
            responses.append(response)
        except Exception as e:
            predictions.append("error")
            responses.append(str(e))
    
    # 多数決
    vote_counts = Counter(predictions)
    predicted_answer = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[predicted_answer] / n_runs
    
    # 正解判定
    is_correct = predicted_answer == correct_answer
    
    result = {
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "response": responses[0],  # 最初の応答を記録
        "ensemble_votes": dict(vote_counts),
        "ensemble_confidence": confidence,
        "ensemble_runs": n_runs,
        "retrieved_docs_count": len(documents),
        "file_name": sample.get('ファイル名', ''),
        "references": sample.get('references', [])
    }
    
    # CoT使用時は複数の推論プロセスも記録
    if use_cot:
        result["reasoning_samples"] = responses[:3]  # 最初の3つを記録
    
    return result


def main():
    parser = argparse.ArgumentParser(description="4択法令データを用いたRAG評価")
    parser.add_argument("--data", type=Path, 
                       default=Path("datasets/lawqa_jp/data/selection.json"),
                       help="4択データセットのパス")
    parser.add_argument("--output", type=Path, 
                       default=Path("results/evaluations/evaluation_results.json"),
                       help="評価結果の出力パス")
    parser.add_argument("--samples", type=int, default=None,
                       help="評価するサンプル数（指定しない場合は全て）")
    parser.add_argument("--no-rag", action="store_true",
                       help="RAGを使用せずLLMのみで評価")
    parser.add_argument("--top-k", type=int, default=3,
                       help="検索する文書数")
    parser.add_argument("--llm-model", type=str, default=None,
                       help="使用するLLMモデル名（指定しない場合は設定ファイルから）")
    parser.add_argument("--no-few-shot", action="store_true",
                       help="Few-shotプロンプトを無効化")
    parser.add_argument("--use-reranker", action="store_true",
                       help="Rerankerを有効化")
    parser.add_argument("--reranker-model", type=str, 
                       default="cross-encoder/ms-marco-MiniLM-L-12-v2",
                       help="Rerankerモデル名")
    parser.add_argument("--rerank-top-n", type=int, default=3,
                       help="Reranker後の文書数")
    parser.add_argument("--ensemble", type=int, default=1,
                       help="アンサンブル推論回数（1=無効、3推奨）")
    parser.add_argument("--use-cot", action="store_true",
                       help="Chain-of-Thought推論を有効化")
    # HFバックエンド用オプション
    parser.add_argument("--llm-backend", type=str, default=None, choices=["ollama", "hf"],
                        help="LLM backend: ollama or hf (Noneならconfig.llm.backendを使用)")
    parser.add_argument("--hf-model-name", type=str, default=None,
                        help="HF backend使用時のモデル名（未指定ならconfig.llm.hf_model_name）")
    parser.add_argument("--hf-lora-path", type=str, default=None,
                        help="HF backend使用時のLoRAパス（未指定ならconfig.llm.lora_path）")
    parser.add_argument("--hf-load-in-4bit", action="store_true",
                        help="HF backendで4bit量子化を強制ON（デフォルトはconfig.llm.load_in_4bit）")
    args = parser.parse_args()
    
    # データセットファイルの存在確認
    if not args.data.exists():
        print(f"Error: Dataset file not found: {args.data}")
        print("\nPlease download the dataset first. See docs/02-SETUP.md for instructions.")
        print("For Heart01 users, you can copy from shared directory:")
        print("  cp -r /home/jovyan/shared/datasets/statutes2025/* ./datasets/")
        sys.exit(1)
    
    # データセット読み込み
    print(f"Loading dataset from {args.data}...")
    try:
        with open(args.data, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {args.data}: {e}")
        sys.exit(1)
    
    # データ構造の検証
    if 'samples' not in data:
        print(f"Error: Invalid dataset format. 'samples' key not found.")
        print(f"Expected format: {{'samples': [...]}}")
        print(f"Please check the dataset file or re-download it.")
        sys.exit(1)
    
    samples = data['samples']
    if not samples:
        print(f"Error: No samples found in dataset.")
        sys.exit(1)
    if args.samples:
        samples = samples[:args.samples]
    
    print(f"Total samples to evaluate: {len(samples)}")
    
    # RAGパイプライン初期化
    print("\nInitializing RAG pipeline...")
    config = load_config()

    # CLIでバックエンド/HF関連が指定されていれば上書き
    if args.llm_backend is not None:
        config.llm.backend = args.llm_backend
    if args.hf_model_name is not None:
        config.llm.hf_model_name = args.hf_model_name
    if args.hf_lora_path is not None:
        config.llm.lora_path = args.hf_lora_path
    if args.hf_load_in_4bit:
        config.llm.load_in_4bit = True
    
    # top_kを引数で上書き
    config.retriever.top_k = args.top_k
    
    # LLMモデル名を引数で上書き
    llm_model = args.llm_model if args.llm_model else config.llm.model_name
    
    # インデックスの存在確認（RAG有効時のみ）
    if not args.no_rag:
        index_path = Path(config.vector_store_path)
        if config.retriever.retriever_type == "vector":
            if not (index_path / "vector").exists():
                print(f"Error: Vector index not found at {index_path / 'vector'}")
                print("\nPlease build the index first:")
                print("  make index")
                sys.exit(1)
        elif config.retriever.retriever_type == "bm25":
            if not (index_path / "bm25").exists():
                print(f"Error: BM25 index not found at {index_path / 'bm25'}")
                print("\nPlease build the index first:")
                print("  make index")
                sys.exit(1)
        elif config.retriever.retriever_type == "hybrid":
            if not (index_path / "vector").exists() or not (index_path / "bm25").exists():
                print(f"Error: Hybrid index not found at {index_path}")
                print("\nPlease build the index first:")
                print("  make index")
                sys.exit(1)
    
    retriever = create_retriever(config) if not args.no_rag else None
    
    # RAG無効時はダミーのRetrieverを使用
    if args.no_rag:
        from unittest.mock import Mock
        retriever = Mock()
        retriever.retrieve = Mock(return_value=[])
    
    # Rerankerの初期化
    reranker = None
    if args.use_reranker:
        print(f"Initializing Reranker: {args.reranker_model}...")
        try:
            reranker = CrossEncoderReranker(model_name=args.reranker_model)
            print("Reranker initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Reranker: {e}")
            print("Continuing without Reranker...")
            reranker = None
    
    # Top-KをReranker使用時に調整
    retriever_top_k = args.top_k if not args.use_reranker else max(args.top_k, args.rerank_top_n * 2)
    
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_provider=config.llm.provider,
        llm_model=llm_model,
        temperature=0.0,  # 4択問題では決定的な回答が望ましい
        reranker=reranker,
        top_k=retriever_top_k,
        rerank_top_n=args.rerank_top_n if args.use_reranker else 5,
        request_timeout=120,  # タイムアウトを120秒に延長
        llm_backend=config.llm.backend,
        hf_model_name=config.llm.hf_model_name,
        lora_path=config.llm.lora_path,
        max_tokens=config.llm.max_tokens,
    )
    
    print(f"RAG Mode: {'Disabled (LLM only)' if args.no_rag else 'Enabled'}")
    print(f"Retriever Type: {config.retriever.retriever_type}")
    print(f"LLM Model: {llm_model}")
    print(f"LLM Backend: {config.llm.backend}")
    if config.llm.backend == "hf":
        print(f"  HF Model: {config.llm.hf_model_name or llm_model}")
        print(f"  LoRA Path: {config.llm.lora_path}")
    print(f"Few-shot Prompt: {'Disabled' if args.no_few_shot else 'Enabled'}")
    print(f"Chain-of-Thought: {'Enabled' if args.use_cot else 'Disabled'}")
    print(f"Reranker: {'Enabled' if args.use_reranker else 'Disabled'}")
    if args.use_reranker:
        print(f"  Model: {args.reranker_model}")
        print(f"  Top-N: {args.rerank_top_n}")
    print(f"Ensemble: {'Enabled ('+str(args.ensemble)+' runs)' if args.ensemble > 1 else 'Disabled'}")
    print(f"Top-K: {retriever_top_k}\n")
    
    # 評価実行
    results = []
    correct_count = 0
    
    for sample in tqdm(samples, desc="Evaluating"):
        if args.ensemble > 1:
            # アンサンブル評価
            result = evaluate_sample_with_ensemble(
                pipeline, sample, 
                n_runs=args.ensemble,
                use_rag=not args.no_rag, 
                use_few_shot=not args.no_few_shot,
                use_cot=args.use_cot
            )
        else:
            # 通常評価
            result = evaluate_sample(
                pipeline, sample, 
                use_rag=not args.no_rag, 
                use_few_shot=not args.no_few_shot,
                use_cot=args.use_cot
            )
        results.append(result)
        
        if result['is_correct']:
            correct_count += 1
    
    # 精度計算
    accuracy = correct_count / len(results) if results else 0
    
    # エラー統計
    error_count = sum(1 for r in results if r['predicted_answer'] == 'error')
    unknown_count = sum(1 for r in results if r['predicted_answer'] == 'unknown')
    timeout_errors = sum(1 for r in results if 'timeout' in r.get('response', '').lower())
    
    # 結果サマリー
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {len(results) - correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nError Analysis:")
    print(f"  Timeout Errors: {timeout_errors}")
    print(f"  Parse Errors (unknown): {unknown_count}")
    print(f"  Other Errors: {error_count}")
    print("="*50)
    
    # 詳細結果を保存
    output_data = {
        "config": {
            "rag_enabled": not args.no_rag,
            "retriever_type": config.retriever.retriever_type,
            "llm_model": llm_model,
            "few_shot_enabled": not args.no_few_shot,
            "cot_enabled": args.use_cot,
            "reranker_enabled": args.use_reranker,
            "reranker_model": args.reranker_model if args.use_reranker else None,
            "rerank_top_n": args.rerank_top_n if args.use_reranker else None,
            "ensemble_runs": args.ensemble if args.ensemble > 1 else None,
            "top_k": retriever_top_k,
            "total_samples": len(results)
        },
        "summary": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(results)
        },
        "results": results
    }
    
    # 出力ファイル名にタイムスタンプを付与して保存
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_path = args.output.with_name(f"{timestamp}_{args.output.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")
    
    # 改善提案
    if timeout_errors > 0:
        print("\n⚠️  Timeout errors detected. Consider:")
        print("  - Increasing --top-k parameter to reduce context size")
        print("  - Using a smaller/faster LLM model")
    if unknown_count > 5:
        print(f"\n⚠️  {unknown_count} parse errors detected. The LLM may not be following instructions.")
        print("  - Check if the LLM model supports Japanese and English instructions")
        print("  - Consider adjusting the prompt template")
    
    # エラーケースの表示
    error_cases = [r for r in results if not r['is_correct']]
    if error_cases:
        print("\n" + "="*50)
        print(f"SAMPLE ERROR CASES (showing up to 10):")
        print("="*50)
        for i, case in enumerate(error_cases[:10], 1):
            print(f"\n[Case {i}] {case['file_name']}")
            print(f"Question: {case['question'][:100]}...")
            print(f"Correct: {case['correct_answer']} | Predicted: {case['predicted_answer']}")
            if case['predicted_answer'] in ['error', 'unknown']:
                print(f"Response: {case['response'][:150]}...")


if __name__ == "__main__":
    main()
