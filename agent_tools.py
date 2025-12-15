"""
エージェントが使用する外部ツール群を定義するモジュール。
安全な計算ロジックと、SerpAPIを利用したGoogle検索機能を提供する。
"""
import ast
import operator
import os
import logging
from langchain_core.tools import tool
from serpapi import GoogleSearch
from typing import Any, Dict

# --------------------------------------------------------------------------
# --- 安全な計算ロジック (ast モジュールを使用) ---
# --------------------------------------------------------------------------

# 許可する演算子をマッピング
ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,  # 単項マイナス (例: -5)
}

def safe_eval_expression(node):
    """
    Pythonの抽象構文木(AST)を安全に評価する再帰関数。
    
    許可されていない関数呼び出しや変数参照を厳しく拒否することで、
    Pythonの組み込み関数 eval() の持つセキュリティリスクを回避する。

    Args:
        node: astモジュールによってパースされたノード。
        
    Returns:
        評価された数値結果。
        
    Raises:
        TypeError: 許可されていない構文や演算子が検出された場合。
    """
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.UnaryOp):
        # 単項演算子 (例: マイナス)
        if type(node.op) not in ALLOWED_OPS:
            raise TypeError(f"許可されていない単項演算子: {type(node.op).__name__}")
        return ALLOWED_OPS[type(node.op)](safe_eval_expression(node.operand))
    elif isinstance(node, ast.BinOp):
        # 二項演算子 (例: +, -, *, /)
        op = type(node.op)
        if op not in ALLOWED_OPS:
            raise TypeError(f"許可されていない演算子: {op.__name__}")
        
        left = safe_eval_expression(node.left)
        right = safe_eval_expression(node.right)
        
        return ALLOWED_OPS[op](left, right)
    elif isinstance(node, ast.Call) or isinstance(node, ast.Name):
        # 関数呼び出しや変数参照を拒否
        raise TypeError(f"関数呼び出しや変数参照は禁止されています: {type(node).__name__}")
    else:
        # その他の危険なノードを拒否
        raise TypeError(f"許可されていない構文: {type(node).__name__}")

@tool
def calculate(expression: str) -> str:
    """
    数式(例: 2 + 2 * (10 / 5))を安全に実行し、結果を返す。
    このツールは、Pythonのeval()を使わず、astモジュールによる安全なパーサーを使用している。
    """
    
    logging.info(f"\n--- [LOG: Calculate Tool Called (Safe Mode)] ---")
    logging.info(f"Input Expression: {expression}")

    try:
        clean_expression = expression.replace('=', '').strip()
        tree = ast.parse(clean_expression, mode='eval')
        result = safe_eval_expression(tree.body)
        
        logging.info(f"Result: {result}")
        return str(result)
    except TypeError as te:
        logging.error(f"安全評価エラー: {te}")
        return f"計算エラー: 許可されていない操作が含まれています。"
    except Exception as e:
        logging.error(f"計算エラー: {e}", exc_info=False) 
        return f"計算エラー: {e}"

# --------------------------------------------------------------------------
# --- Google Search ツール ---
# --------------------------------------------------------------------------

@tool
def google_search(query: str) -> str:
    """
    SerpAPIを利用してインターネットで最新の情報を検索し、最も関連性の高いスニペット（とURL）を返す。
    
    Args:
        query: Google検索に渡すクエリ文字列。
        
    Returns:
        上位3件の検索結果（アンサーボックス、スニペット、URL）を結合した文字列。
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    
    if not api_key:
         logging.warning("SerpAPIキーが設定されていません。")
         return "SerpAPIキーが設定されていません。検索を実行できません。"

    params: Dict[str, Any] = {
        "api_key": api_key,
        "engine": "google",
        "q": query,
        "gl": "jp",
        "hl": "ja",
    }
    
    logging.info("\n--- [LOG: SerpAPI Request Details] ---")

    try:
        search = GoogleSearch(params)
        raw_results = search.get_dict()
        
        combined_results = []
        
        # アンサーボックス、知識グラフを優先
        if raw_results.get('answer_box', {}).get('snippet'):
            combined_results.insert(0, f"アンサーボックス: {raw_results['answer_box']['snippet']}")
        if raw_results.get('knowledge_graph', {}).get('snippet'):
            combined_results.insert(0, f"知識グラフ: {raw_results['knowledge_graph']['snippet']}")
        
        # オーガニック検索結果
        for result in raw_results.get('organic_results', []):
            title = result.get('title')
            snippet = result.get('snippet')
            link = result.get('link')
            if title and link:
                combined_results.append(f"タイトル: {title} | スニペット: {snippet} | URL: {link}")
        
        if combined_results:
            result = " ||| ".join(combined_results[:3])
            logging.info(f"--- [LOG: Google Search Tool Return] ---")
            return result
            
        return "検索結果が見つかりませんでした。"
        
    except Exception as e:
        logging.error(f"SerpAPI通信中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return f"SerpAPI通信中に予期せぬエラーが発生しました: {e}"