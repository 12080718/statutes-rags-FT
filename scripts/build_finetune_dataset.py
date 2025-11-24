#!/usr/bin/env python3
"""
法令4択問題（lawqa_jp 等）からファインチューニング用の JSONL データを生成するスクリプト。

主な役割:
- lawqa_jp の selection.json を読み込み、共通フォーマットに正規化する
- RAGPipeline で条文コンテキストを取得（任意）
- prompts.py でプロンプトを構築し、{"input","output","meta"} 形式のレコードを組み立てる
- JSONL として書き出す
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union, Literal

from dotenv import load_dotenv

# リポジトリルートを import パスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.rag_config import load_config
from app.core.prompts import build_mc_prompt_cot, build_mc_prompt_direct
from app.retrieval.base import Document


# プロジェクトルートの .env を読み込む
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)


def parse_choices(raw: Any) -> Dict[str, str]:
    """
    lawqa形式の「選択肢」を a/b/c/d 辞書に整形する。

    Args:
        raw: 文字列（各行が "a ..."）またはすでに dict の場合を想定。

    Returns:
        {"a": str, "b": str, "c": str, "d": str} の辞書（足りないキーは空文字）。
    """
    if isinstance(raw, dict):
        # 既に dict の場合は小文字キーで取り出す
        return {k.lower(): str(v) for k, v in raw.items() if k.lower() in {"a", "b", "c", "d"}}

    choices: Dict[str, str] = {}
    if isinstance(raw, str):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # 先頭の a/b/c/d を抽出（例: "a 〜", "b. 〜", "c) 〜"）
            prefix = line[:1].lower()
            if prefix in {"a", "b", "c", "d"}:
                # 区切り文字を除去
                text = line[1:].lstrip(").．.、　 ").strip()
                choices[prefix] = text
    # 足りないキーを空文字で埋める
    for key in ["a", "b", "c", "d"]:
        choices.setdefault(key, "")
    return choices


def load_lawqa(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    lawqa_jp の selection.json を読み込み、共通フォーマットに正規化する。

    Args:
        path: lawqa_jp の JSON へのパス。

    Returns:
        以下のキーを持つ辞書のリスト:
        - id: 一意なID（ファイル名または連番）
        - question: 問題文
        - choices: {"a","b","c","d"} の辞書
        - correct: 正答（小文字1文字）
        - source_file: 元データのファイル名（あれば）
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples_raw = data.get("samples")
    if not isinstance(samples_raw, list):
        raise ValueError("lawqa_jp データの形式が不正です: 'samples' が見つかりません。")

    normalized: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples_raw):
        question = sample.get("問題文") or sample.get("question")
        if not question:
            raise ValueError(f"{path}: {idx} 行目で '問題文' が見つかりません。")

        raw_choices = sample.get("選択肢") or sample.get("choices") or {}
        choices = parse_choices(raw_choices)

        correct = (sample.get("output") or sample.get("correct") or "").strip().lower()
        if correct not in {"a", "b", "c", "d"}:
            raise ValueError(f"{path}: {idx} 行目の正答が a〜d ではありません: {correct!r}")

        sample_id = sample.get("id") or sample.get("ファイル名") or f"lawqa-{idx:05d}"
        source_file = sample.get("ファイル名") or sample.get("source_file") or path.name

        normalized.append(
            {
                "id": str(sample_id),
                "question": str(question),
                "choices": choices,
                "correct": correct,
                "source_file": source_file,
                "dataset": "lawqa_jp",
            }
        )

    return normalized


def build_context(pipeline: Any, question: str, top_k: int) -> Tuple[List[Document], str]:
    """
    RAGPipeline で質問に対する条文コンテキストを取得する。

    Args:
        pipeline: RAGPipeline インスタンス。
        question: 質問文。
        top_k: 取得するチャンク数。

    Returns:
        (documents, context) のタプル。context は format_context で整形された文字列。
    """
    # top_k を一時的に上書きして検索
    original_top_k = getattr(pipeline, "top_k", top_k)
    pipeline.top_k = top_k
    documents = pipeline.retrieve_documents(question)
    pipeline.top_k = original_top_k
    context = pipeline.format_context(documents) if documents else ""
    return documents, context


def make_instance(
    sample: Dict[str, Any],
    context: str,
    *,
    mode: Literal["direct", "cot"],
    few_shot: bool = False,
    top_k: int = 0,
    retriever_type: str = "hybrid",
) -> Dict[str, Any]:
    """
    1問分のサンプルとコンテキストから JSONL 1レコードを生成する。

    Args:
        sample: load_lawqa で正規化したサンプル。
        context: 条文コンテキスト文字列（空文字ならフォールバック扱い）。
        mode: "direct" か "cot" のいずれか。
        few_shot: few-shot 例を付与するか。
        top_k: 取得に使ったチャンク数（メタ情報用）。
        retriever_type: 使用リトリーバ種別（メタ情報用）。

    Returns:
        {"input": str, "output": str, "meta": Dict[str, Any]} の辞書。
    """
    question = sample["question"]
    choices = sample["choices"]
    correct = sample["correct"]
    dataset = sample.get("dataset", "lawqa_jp")
    source_file = sample.get("source_file", "")

    # 入力プロンプトの生成
    if mode == "cot":
        prompt = build_mc_prompt_cot(question, choices, context, style="compact")
        output = f"Reasoning: 条文に基づき選択肢を検討した結果、選択肢{correct}が最も適切と判断。\nAnswer: {correct}"
    else:
        prompt = build_mc_prompt_direct(question, choices, context, few_shot=few_shot)
        output = correct

    use_context = bool(context.strip())
    record = {
        "input": prompt,
        "output": output,
        "meta": {
            "id": sample["id"],
            "dataset": dataset,
            "mode": mode,
            "correct": correct,
            "use_context": use_context,
            "context_present": use_context,
            "top_k": top_k,
            "retriever_type": retriever_type,
            "question": question,
            "choices": choices,
            "source_file": source_file,
        },
    }
    return record


def build_dataset(
    samples: List[Dict[str, Any]],
    pipeline: Optional[Any],
    *,
    mode: Literal["direct", "cot"] = "direct",
    top_k: int = 3,
    few_shot: bool = False,
    use_rag: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    サンプル全体を処理して JSONL レコードを順次生成する。

    Args:
        samples: load_lawqa で正規化したサンプル一覧。
        pipeline: RAGPipeline（use_rag=False の場合は None 可）。
        mode: "direct" または "cot"。
        top_k: 取得するチャンク数。
        few_shot: few-shot 例を付与するか。
        use_rag: RAG を有効にするか。

    Yields:
        make_instance が生成したレコード。
    """
    for sample in samples:
        if use_rag and pipeline:
            _, context = build_context(pipeline, sample["question"], top_k)
            retriever_type = getattr(getattr(pipeline, "retriever", None), "__class__", type("x", (), {})).__name__.lower()
        else:
            context = ""
            retriever_type = "none"

        yield make_instance(
            sample,
            context,
            mode=mode,
            few_shot=few_shot,
            top_k=top_k if use_rag else 0,
            retriever_type=retriever_type,
        )


def save_jsonl(records: Iterable[Dict[str, Any]], path: Union[str, Path]) -> None:
    """
    レコードの反復可能オブジェクトを JSONL 形式で保存する。

    Args:
        records: {"input","output","meta"} を含む辞書の反復可能。
        path: 出力先パス。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_retriever(config):
    """
    RAG 用 Retriever を設定に従って構築する。

    Args:
        config: RAGConfig インスタンス。

    Returns:
        Vector/BM25/Hybrid のいずれかの Retriever インスタンス。
    """
    # 遅延インポートで依存ライブラリ未導入環境でも no-rag 動作を許容
    from app.retrieval.vector_retriever import VectorRetriever
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.hybrid_retriever import HybridRetriever

    retriever_type = config.retriever.retriever_type
    index_path = Path(config.vector_store_path)

    if retriever_type == "vector":
        return VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda,
            mmr_fetch_k_max=config.retriever.mmr_fetch_k_max,
        )
    elif retriever_type == "bm25":
        return BM25Retriever(
            index_path=str(index_path / "bm25"),
            tokenizer=config.retriever.bm25_tokenizer,
        )
    else:
        vector_retriever = VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda,
            mmr_fetch_k_max=config.retriever.mmr_fetch_k_max,
        )
        bm25_retriever = BM25Retriever(
            index_path=str(index_path / "bm25"),
            tokenizer=config.retriever.bm25_tokenizer,
        )
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fusion_method=config.retriever.fusion_method,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight,
            rrf_k=config.retriever.rrf_k,
            fetch_k_multiplier=config.retriever.fetch_k_multiplier,
        )


def main() -> None:
    """
    CLI エントリポイント。
    lawqa_jp データからファインチューニング用 JSONL を生成する。
    """
    parser = argparse.ArgumentParser(description="lawqa_jp 4択問題からFT用JSONLを生成する")
    parser.add_argument(
        "--lawqa-path",
        type=Path,
        default=Path("datasets/lawqa_jp/data/selection.json"),
        help="lawqa_jp の selection.json パス",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/finetune/ft_dataset.jsonl"),
        help="出力する JSONL のパス",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "cot"],
        default="direct",
        help="プロンプトのモード（direct: 1文字回答, cot: 推論付き回答）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="コンテキストとして取得するチャンク数",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="few-shot 例をプロンプトに含める",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="RAG を使わずにコンテキストなしで生成する",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="処理するサンプル数（デバッグ用、指定しなければ全件）",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Reranker を有効化する（利用可能な環境のみ）",
    )
    args = parser.parse_args()

    samples = load_lawqa(args.lawqa_path)
    if args.samples:
        samples = samples[: args.samples]

    use_rag = not args.no_rag

    pipeline: Optional[RAGPipeline] = None
    retriever_type = "none"
    if use_rag:
        config = load_config()
        retriever_type = config.retriever.retriever_type
        retriever = create_retriever(config)
        reranker = None
        if args.use_reranker:
            try:
                from app.retrieval.reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker(model_name=config.reranker.model_name)
            except Exception as e:
                print(f"Reranker 初期化に失敗しました: {e}")
                reranker = None

        try:
            from app.retrieval.rag_pipeline import RAGPipeline
        except ModuleNotFoundError as e:
            raise SystemExit(f"RAGPipeline の依存関係が不足しています。必要に応じて `pip install langchain-community` などを実行してください: {e}")

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_provider=config.llm.provider,
            llm_model=config.llm.model_name,
            temperature=config.llm.temperature,
            reranker=reranker,
            top_k=max(args.top_k, config.retriever.top_k),
            rerank_top_n=config.reranker.top_n if reranker else 5,
        )
    records = list(
        build_dataset(
            samples,
            pipeline,
            mode=args.mode,
            top_k=args.top_k,
            few_shot=args.few_shot,
            use_rag=use_rag,
        )
    )
    save_jsonl(records, args.output_path)

    print("====== 生成結果 ======")
    print(f"入力ファイル: {args.lawqa_path}")
    print(f"出力ファイル: {args.output_path}")
    print(f"モード: {args.mode}")
    print(f"RAG使用: {use_rag}")
    print(f"件数: {len(records)}")
    print("======================")


if __name__ == "__main__":
    main()
