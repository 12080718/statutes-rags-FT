"""
複数の評価・ファインチューニング処理で共通利用する4択問題プロンプトのテンプレート群。

日本法令の4択問題に対するプロンプト組み立てを一元化し、
`scripts/evaluate_multiple_choice.py` や将来の学習データ生成で再利用できるようにする。
"""
from __future__ import annotations

from typing import Dict, Literal, Optional

__all__ = [
    "build_mc_prompt_direct",
    "build_mc_prompt_cot",
]


def _format_choices(choices: Dict[str, str]) -> str:
    """
    4択選択肢の辞書を人が読めるブロック文字列に整形する。

    Args:
        choices: a/b/c/d をキー、選択肢本文を値とする辞書。

    Returns:
        各行の先頭に選択肢のアルファベットが付与された文字列。
    """
    ordered_keys = ["a", "b", "c", "d"]
    lines = []
    for key in ordered_keys:
        value = choices.get(key, "")
        lines.append(f"{key} {value}")
    return "\n".join(lines)


def build_mc_prompt_direct(
    question: str,
    choices: Dict[str, str],
    context: Optional[str] = None,
    *,
    few_shot: bool = True,
) -> str:
    """
    法令4択問題向けの直接回答プロンプト（CoTなし）を生成する。

    モデルに単一文字（a/b/c/d）のみを出力させることを目的とした形式。
    few-shot 例を前置して挙動を安定させることもでき、`few_shot` で制御する。

    Args:
        question: 日本語の4択問題文。
        choices: ``"a"``〜``"d"`` をキーとする選択肢辞書。
        context: 回答の根拠とする取得済み法令テキスト。``None`` の場合はフォールバック文を挿入。
        few_shot: True の場合、単一文字で回答する例を先頭に付与して安定化させる。

    Returns:
        LLMにそのまま渡せるプロンプト文字列。
    """
    context_block = context or "法令条文が提供されていません。既有知識に基づいて回答してください。"
    choices_block = _format_choices(choices)

    few_shot_block = ""
    if few_shot:
        few_shot_block = (
            "例（正しいものを選ぶ）:\n"
            "【法令条文】\n"
            "民法第90条: 公の秩序又は善良の風俗に反する法律行為は、無効とする。\n"
            "【質問】公序良俗に反する法律行為はどう扱われるか。\n"
            "【選択肢】\n"
            "a 取り消すことができる\n"
            "b 無効である\n"
            "c 公序良俗に反しても有効である\n"
            "d 無効か有効か裁判所が決める\n"
            "【回答】b\n\n"
        )

    prompt = (
        "あなたは日本の法律に精通したリーガルアシスタントです。"
        "以下の法令条文に基づいて4択問題に回答してください。\n\n"
        f"{few_shot_block}"
        "【法令条文】\n"
        f"{context_block}\n\n"
        "【質問】\n"
        f"{question}\n\n"
        "【選択肢】\n"
        f"{choices_block}\n\n"
        "【回答指示】\n"
        "- 上記の法令条文の内容のみを根拠に選んでください。\n"
        "- 回答は a, b, c, d のいずれか1文字のみ。\n"
        "- 理由や説明は書かず、1文字だけ出力してください。\n\n"
        "回答（1文字のみ）: "
    )
    return prompt


def build_mc_prompt_cot(
    question: str,
    choices: Dict[str, str],
    context: Optional[str] = None,
    *,
    style: Literal["compact", "detailed"] = "compact",
) -> str:
    """
    法令4択問題向けの Chain-of-Thought (CoT) プロンプトを生成する。

    ステップごとの推論を促したうえで、最終的に単一文字を出力させる形式。
    推論過程を観察したい場合や、学習データ作成で理由文を収集したい場合に有用。

    Args:
        question: 日本語の4択問題文。
        choices: ``"a"``〜``"d"`` をキーとする選択肢辞書。
        context: 回答の根拠とする取得済み法令テキスト。``None`` の場合はフォールバック文を挿入。
        style: 推論手順の指示の詳細度を指定。
            - ``"compact"``: 簡潔な番号付きステップ。
            - ``"detailed"``: 各ステップをやや丁寧に指示。

    Returns:
        LLMに渡すプロンプト文字列。推論後に ``Answer: <letter>`` を返すことを想定。
    """
    context_block = context or "法令条文が提供されていません。既有知識に基づいて回答してください。"
    choices_block = _format_choices(choices)

    if style == "detailed":
        reasoning_instructions = (
            "1. 質問が「正しいもの」か「誤っているもの」かを確認する。\n"
            "2. 条文から要件・結論を抜き出し、各選択肢と突き合わせる。\n"
            "3. 合致する/しない理由を簡潔に述べ、最終的に一つの選択肢を選ぶ。\n"
        )
    else:
        reasoning_instructions = (
            "1. 条文の要点をまとめる。\n"
            "2. 各選択肢を条文に照らして妥当性を判定する。\n"
            "3. 最も適切な選択肢を一つだけ選ぶ。\n"
        )

    prompt = (
        "あなたは日本の法律に精通したリーガルアシスタントです。"
        "条文を根拠にステップごとに考え、最終的な選択肢を1文字で回答してください。\n\n"
        "【法令条文】\n"
        f"{context_block}\n\n"
        "【質問】\n"
        f"{question}\n\n"
        "【選択肢】\n"
        f"{choices_block}\n\n"
        "【推論手順】\n"
        f"{reasoning_instructions}\n"
        "出力形式:\n"
        "Reasoning: <ステップごとの考察を日本語で>\n"
        "Answer: <a/b/c/d の1文字>\n"
    )
    return prompt
