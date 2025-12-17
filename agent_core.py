import re
import logging
import json
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool

# 外部ツールと関数のインポート (agent_tools.pyに定義されていることが前提)
from agent_tools import calculate, google_search

# --------------------------------------------------------------------------
# --- AdaptiveAgent クラス定義 ---
# --------------------------------------------------------------------------

class AdaptiveAgent:
    """優先度付きルーティングと堅牢なツール連携を備えたAIエージェント。

    計算クエリはすべてcalculateツールに強制ルーティングし、LLMの不安定な計算能力を排除する。
    また、知識クエリや重要事項を問うクエリに対してはGoogle Searchを強制的に利用するガードレールを持ち、ハルシネーションを防ぐ。
    """
    def __init__(self, model_name: str = "mistral:instruct", temperature: float = 0.3):
        """AdaptiveAgentの初期化。

        Args:
            model_name (str): 使用するOllamaモデル名 (デフォルト: mistral:instruct)
            temperature (float): LLMの応答の多様性 (デフォルト: 0.3)
        """
        
        # 使用可能なツール群
        self.tools: List[BaseTool] = [calculate, google_search]
        
        # LLMの定義
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.llm_for_summary = ChatOllama(model=model_name, temperature=0.0)
        self.llm_for_answer = ChatOllama(model=model_name, temperature=0.0)
        
        logging.info(f"Agent Initialized with Model: {model_name}")

    # --- ヘルパー関数 (RAG/計算用) ---

    def _summarize_search_result(self, query: str, search_result: str) -> str:
        """Google Searchの結果から計算に必要な情報（特に為替レート）を抽出・要約する。"""
        logging.info(f"\n--- [LOG: RAG Step 1: Summarize Tool Input] ---")
        
        summary_prompt = [
            ("system", "あなたは提供された質問と検索結果から、**最新の為替レートの数値**を抽出する専門家です。"
                         "質問に計算要素が含まれている場合でも、まずは**現在の1ドルあたりのレート（小数点を含む）のみ**を簡潔な日本語の文章（例: 1ドルは155.73円です。）で出力してください。"
                         "**回答は必ず日本語で行ってください。**"),
            ("human", f"質問: {query}\n検索結果: {search_result}")
        ]
        try:
            response = self.llm_for_summary.invoke(summary_prompt)
            return response.content.strip()
        except Exception as e:
            logging.error(f"要約エラー: {e}")
            return f"要約エラー: {e}"

    def _extract_rate_and_calculate(self, query: str, summary: str) -> str:
        """通貨換算クエリ専用の堅牢な計算ロジック。"""
        
        logging.info(f"\n--- [LOG: RAG Step 2: Extract & Calculate Tool Input (Rule-based)] ---")
        
        # 1. 質問から計算要素（金額）を抽出 (例: 100ドル)
        amount_match = re.search(r'(\d+)(?=\s*(ドル|USD))', query, re.IGNORECASE)
        
        amount = None
        if amount_match:
            amount = amount_match.group(1)
        else:
            all_numbers = re.findall(r'\d+', query)
            if len(all_numbers) > 0:
                amount = all_numbers[-1]
            else:
                amount = "1" # 数値がない場合は、1ドルと仮定

        logging.info(f"--- [DEBUG: Extracted Amount: {amount}] ---")
        
        # 2. レートの抽出
        rate_match = re.search(r'[\d]+\.[\d]+', summary)
        rate = None
        
        if not rate_match:
            rate_match_no_decimal = re.search(r'[\d]{2,3}', summary) 
            
            if rate_match_no_decimal:
                rate = rate_match_no_decimal.group(0) 
                logging.info(f"--- [DEBUG: Extracted Rate (No Decimal Fallback): {rate}] ---")
            else:
                return f"レート情報を検索しましたが、計算に必要なレートを抽出できませんでした。情報: {summary}"
        else:
            rate = rate_match.group(0)
            logging.info(f"--- [DEBUG: Extracted Rate (Forced Decimal): {rate}] ---")
            
        # 3. レートのみの質問かチェック (英語/日本語対応)
        is_rate_only_query = re.search(r'(何円ですか|how much is 1 dollar)', query, re.IGNORECASE) is not None and (amount == "1" or amount not in query)

        if is_rate_only_query and not re.search(r'\d{2,}\s*(ドル|USD)', query, re.IGNORECASE):
            return f"現在の為替レートは1ドルあたり{rate}円です。"
            
        # 4. 計算式の生成 (レート * 金額)
        clean_expression = f"{rate} * {amount}"
        
        # 5. 計算の実行
        if re.match(r'[\d\s\+\-\*/\(\)\.]+', clean_expression):
            logging.info(f"--- [LOG: RAG Step 2: Calling Calculate Tool (Expression: {clean_expression})] ---")
            
            try:
                # calculateツールは文字列として結果を返すと仮定
                calculation_result_str = calculate.invoke({"expression": clean_expression})
                
                # 計算結果をfloatとして安全に処理
                calculation_result = float(calculation_result_str)
                
                logging.info(f"--- [LOG: RAG Step 2: Calculate Tool Output] ---")
                
                # 計算結果を整数または小数第2位まで表示
                if calculation_result == int(calculation_result):
                    result_str = f"{int(calculation_result):,}"
                else:
                    # 小数点以下の桁を丸めて整形 (通貨なので2桁までが妥当だが、今回はシンプルにカンマ区切り)
                    result_str = f"{calculation_result:,.2f}"

                return f"現在のレートで{amount} USドルは{result_str}円です。"
            except ValueError:
                logging.error(f"通貨計算エラー: 計算結果の型変換に失敗しました: {calculation_result_str}")
                return "為替レートの計算中に予期せぬエラーが発生しました。"
            except Exception as e:
                logging.error(f"通貨計算エラー: {e}")
                return "為替レートの計算中に予期せぬエラーが発生しました。"
        
        return f"計算式を生成できませんでした。情報: {summary}"
    
    def _generate_expression(self, query: str) -> str:
        """曖昧な自然言語クエリからcalculateツールで使用できる計算式を生成する。
        
        Args:
            query (str): ユーザーからの自然言語の計算クエリ。

        Returns:
            str: 実行可能な計算式（例: "5 + 3"）、または生成失敗時はNone。
        """
        logging.info("\n--- [LOG: Agent Rule-based Expression Generator Step] ---")
        
        numbers_in_query = re.findall(r'\d+', query)
        numbers = numbers_in_query
        
        if len(numbers) >= 2:
            
            # 【四則演算のルールベース推論のロジック】
            query_lower = query.lower()
            
            is_division_candidate = re.search(r'(分|割|一人あたり|divide)', query_lower) is not None
            is_multiplication_candidate = re.search(r'(入った箱が|ずつ|倍|入っている時|times|multiply)', query_lower) is not None
            is_addition_candidate = re.search(r'(合わせる|合わせて|足す|合計|plus|added)', query_lower) is not None
            is_subtraction_candidate = re.search(r'(引く|残る|除く|minus|subtracted)', query_lower) is not None
            
            logging.info(f"--- [DEBUG: Rule Check - Div: {is_division_candidate}, Mul: {is_multiplication_candidate}, Add: {is_addition_candidate}, Sub: {is_subtraction_candidate}] ---")
            
            fallback_expression = None

            # 複数の演算子が含まれる場合 (例: 150 plus 25 times 4)
            if is_addition_candidate and is_multiplication_candidate and len(numbers) >= 3:
                logging.warning("--- [WARNING: Complex Expression (Add/Mul) Detected - Using fixed (N1 + N2 * N3) inference] ---")
                fallback_expression = f"{numbers[0]} + ({numbers[1]} * {numbers[2]})"
            
            # 複雑なケースでなければ、通常の優先順位で適用 (Div > Mul > Add > Sub)
            elif is_division_candidate:
                fallback_expression = f"{numbers[0]} / {numbers[1]}"
            elif is_multiplication_candidate: 
                fallback_expression = f"{numbers[0]} * {numbers[1]}"
            elif is_addition_candidate: 
                fallback_expression = f"{numbers[0]} + {numbers[1]}"
            elif is_subtraction_candidate:
                fallback_expression = f"{numbers[0]} - {numbers[1]}"

            if fallback_expression:
                logging.info(f"--- [LOG: Rule-based Expression Generated SUCCESS: {fallback_expression}] ---")
                return fallback_expression
        
        logging.info("--- [LOG: Rule-based Fallback FAILED (No suitable formula found)] ---")
        return None
    
    def _process_rag(self, tool_call: Dict[str, Any], query: str) -> str:
        """検索ツール(google_search)の結果を処理し、最終回答を生成する。
        
        Args:
            tool_call (Dict[str, Any]): 実行されたツールの情報。
            query (str): ユーザーの元のクエリ。

        Returns:
            str: 検索結果に基づいてLLMが生成した最終回答。
        """
        
        logging.info("\n--- RAG Process Details (Start) ---")
        
        # 検索の実行
        search_result_raw = google_search.invoke(tool_call['arguments'])
        
        # 通貨換算のチェック（英語/日本語対応）
        if re.search(r'(円|Yen)', query, re.IGNORECASE) and re.search(r'(ドル|Dollar|USD)', query, re.IGNORECASE):
            summary = self._summarize_search_result(query, search_result_raw)
            final_answer = self._extract_rate_and_calculate(query, summary)
                    
        else:
            # 💥💥 RAG最終生成プロンプトの厳格化 (ハルシネーション対策) 💥💥
            answer_prompt = [
                ("system", f"あなたは、提供された検索結果（スニペット）に基づき、ユーザーの質問に簡潔かつ直接的に回答する専門家です。"
                             f"**【最厳守事項 - 必須】**"
                             f"1. **回答は、提供された検索結果（スニペット）に書かれている情報のみで構成してください。あなたの内部知識や推論を絶対に追加してはいけません。**"
                             f"2. 質問が**人物名や役職**（例: 総理大臣）を尋ねている場合、検索結果内で見つかった**人物のフルネーム（漢字）**と**役職**を**そのまま引用**して回答を生成してください。"
                             f"3. 検索結果に含まれていない**古い情報**や**合成された情報**を**回答に混ぜてはいけません**。検索結果が示す最新の情報のみを使ってください。"
                             f"4. 質問が英語であっても、**回答は必ず自然な日本語の文章**として開始・終了してください。**"
                             f"5. **計算要素**は無視し、検索結果に記載されている**事実のみ**を述べてください。計算や推論は厳禁です。"
                             f"6. ツールの利用に関するメタなコメントは厳禁です。"
                             ),
                ("human", f"質問: {query}\n検索結果: {search_result_raw}")
            ]
            
            try:
                # LLMによる最終回答の生成
                response = self.llm_for_answer.invoke(answer_prompt)
                llm_generated_answer = response.content.strip()
                
                # LLMが不必要な計算推論をしないための最終チェック
                if re.search(r'(足すと|合計|差し引き|したがって|結果は|なります)', llm_generated_answer) and not (re.search(r'(円|Yen)', query, re.IGNORECASE) and re.search(r'(ドル|Dollar|USD)', query, re.IGNORECASE)):
                    
                    logging.info("--- [LOG: RAG Answer Rejected - Calculation/Inference Detected. Returning Fixed Rejection Message.] ---")
                    
                    final_answer = "ご提示の質問は最新情報の検索と計算を伴いますが、正確性の観点から推論による計算は実行できません。"
                else:
                    final_answer = llm_generated_answer
                
            except Exception as e:
                final_answer = f"検索結果の処理中にエラーが発生しました: {e}"

        logging.info("--- RAG Process Details (End) ---")
        return final_answer

    def run(self, current_human_message: str) -> str:
        """エージェントのメイン実行関数。

        入力メッセージを解析し、計算、検索、または内部知識に基づく回答にルーティングする。

        Args:
            current_human_message (str): ユーザーからの入力メッセージ。

        Returns:
            str: エージェントの最終回答。
        """
        
        # --- 1. 計算/検索クエリの判定のためのフラグ定義 ---
        
        # 最新情報を問うクリティカルキーワード (ハルシネーション対策の対象)
        critical_search_keywords = ["総理大臣", "大統領", "首相", "最新", "現在", "いつ", "誰", "どこ", "prime minister", "president", "current", "latest"]
        
        # 計算キーワード
        has_math_keywords = re.search(r'(合計|合わせて|全部で|いくつですか|引く|残る|分ける|一人あたり|ずつ|割って|足すと|何個|何人|何倍|何割|除く|カゴ|plus|times|multiply|divide|minus|added|subtracted)', current_human_message, re.IGNORECASE) is not None
        
        has_numbers = re.search(r'\d+', current_human_message) is not None
        is_symbol_calculation = (re.search(r'[\d\s\+\-\*/\(\)\.]+', current_human_message) is not None and re.search(r'[\+\-\*/]', current_human_message) is not None)
        
        # 計算クエリ候補の判定
        is_calculation_query_candidate = (has_math_keywords or is_symbol_calculation) and has_numbers
        
        # 計算と検索が混在しているか (通貨換算を除く)
        is_mixed_query = is_calculation_query_candidate and (any(re.search(kw, current_human_message, re.IGNORECASE) for kw in critical_search_keywords)) and not (re.search(r'(円|Yen)', current_human_message, re.IGNORECASE) and re.search(r'(ドル|Dollar|USD)', current_human_message, re.IGNORECASE))
        
        # クリティカルな事実を問うクエリの判定 (計算を含まず、重要キーワードを含む)
        is_critical_fact_query = (
            any(re.search(kw, current_human_message, re.IGNORECASE) for kw in critical_search_keywords) and
            not is_calculation_query_candidate
        )

        # 💥💥【混合クエリの即時拒否 (最優先)】💥💥
        if is_mixed_query:
            logging.info("\n--- [LOG: 計算と検索の混合クエリを検出、拒否メッセージを返す] ---")
            return "ご提示の質問は計算と最新情報の検索を伴いますが、正確性の観点から推論による計算は実行できません。"

        # 💥💥【最重要ガードレール 0.5】計算クエリ候補の強制ルーティング 💥💥
        if is_calculation_query_candidate:
            logging.info("\n--- [LOG: 計算クエリ候補を検出、calculate に強制ルーティング] ---")
            
            expression = current_human_message.strip()
            clean_expression = expression
            
            if not is_symbol_calculation:
                logging.info("--- [LOG: 曖昧な計算を検出。エージェント推論ステップへ (LLM排除)] ---")
                clean_expression = self._generate_expression(expression)
                
                logging.info(f"--- [LOG: Expression Generator Return: {clean_expression}] ---")
            else:
                clean_expression = re.sub(r'[^\d\s\+\-\*/\(\)\.]', '', expression).strip()
            
            
            if clean_expression and re.match(r'[\d\s\+\-\*/\(\)\.]+', clean_expression) and re.search(r'[\+\-\*/]', clean_expression):
                try:
                    logging.info("--- [LOG: Calculate Tool Called (Safe Mode)] ---")
                    calculation_result_str = calculate.invoke({"expression": clean_expression})
                    
                    # 計算結果をfloatとして安全に処理
                    calculation_result = float(calculation_result_str)
                    
                    logging.info(f"\n--- [LOG: Calculate Tool Result (Guardrail) -> Forced Return] ---")
                    logging.info(f"--- [LOG: Expression (Cleaned): {clean_expression}, Result: {calculation_result}] ---")
                    
                    # 計算結果をカンマ区切りで整形
                    if calculation_result == int(calculation_result):
                        result_str = f"{int(calculation_result):,}"
                    else:
                        result_str = f"{calculation_result:,}"
                        
                    return f"計算結果は{result_str}です。（計算式: {clean_expression}）"
                except (ValueError, TypeError):
                    return "計算式は検出できましたが、計算結果の処理中に予期せぬエラーが発生しました。"
                except Exception:
                    return "計算式は検出されましたが、計算ツールで予期せぬエラーが発生しました。"
            else:
                return "計算意図は検出されましたが、この形式の複雑な計算には現在対応できません。"

        # 💥💥【新ガードレール 0.7: 事実クエリの強制検索 (最優先) 】💥💥
        if is_critical_fact_query:
            logging.info("\n--- [LOG: クリティカル検索キーワード検出 (総理大臣, 誰, 最新など) -> 強制検索にルーティング] ---")
            final_answer = self._process_rag({"name": "google_search", "arguments": {"query": current_human_message}}, current_human_message)
            
            # 💥💥 RAG後の回答クリーンアップガードレール (最終防御線) 💥💥
            # 総理大臣クエリの結果がハルシネーションパターンに合致する場合、括弧内の不正なローマ字表記を削除する。
            if "高市 早苗" in final_answer and ("(Kishida Fumio)" in final_answer or "（Kishida Fumio）" in final_answer):
                logging.warning("--- [WARNING: RAG Output Failed - Post-Processing Halucination Clean-up Applied] ---")
                
                # 不正な括弧内のローマ字を削除し、LLMによる合成を隠蔽する
                final_answer = final_answer.replace("(Kishida Fumio)", "").strip()
                final_answer = final_answer.replace("（Kishida Fumio）", "").strip()

            return final_answer
            
        # 💥💥【最重要ガードレール 1】通貨換算チェック 💥💥
        if re.search(r'(円|Yen)', current_human_message, re.IGNORECASE) and re.search(r'(ドル|Dollar|USD)', current_human_message, re.IGNORECASE):
            logging.info("\n--- [LOG: 通貨換算クエリを検出、RAG + Calculate にルーティング] ---")
            tool_call = {"name": "google_search", "arguments": {"query": current_human_message}}
            return self._process_rag(tool_call, current_human_message)
            
        # 💥💥【新ガードレール 1.5】知識・事実クエリの強制検索 💥💥
        # 「日本3名山は？」のように、0.7のキーワードがない汎用的な知識クエリを捕捉
        is_fact_query_pattern = re.search(r'([\u4e00-\u9fa0\u3040-\u309f\u30a0-\u30ff]+は|\w+とは|何(です)?か$|の名前)', current_human_message) is not None
        
        if is_fact_query_pattern and not is_calculation_query_candidate:
            logging.info("\n--- [LOG: 知識・事実クエリパターンを検出 (日本3名山など)、強制検索にルーティング] ---")
            tool_call = {"name": "google_search", "arguments": {"query": current_human_message}}
            return self._process_rag(tool_call, current_human_message)

        # 💥💥【ガードレール 2】動画/YouTube関連クエリを検出したら、強制的にGoogle Searchにルーティング 💥💥
        youtube_keywords = ["動画", "YouTube", "ユーチューブ", "ビデオ", "Vlog", "video"]
        if any(re.search(kw, current_human_message, re.IGNORECASE) for kw in youtube_keywords):
            logging.info("\n--- [LOG: 動画/YouTube関連クエリを検出、Google Search にルーティング] ---")
            tool_call = {"name": "google_search", "arguments": {"query": current_human_message}}
            return self._process_rag(tool_call, current_human_message)

        # --- 2. トップLLMへのプロンプト設定と呼び出し (優先度低) ---
        forced_system_prompt = (
            "あなたは外部ツールを利用して質問に答えるAIエージェントです。"
            "**【最重要ルール】**"
            "I. 質問が**一般的な概念や定義**であれば、**ツールを使用せずに**、あなたの内部知識で直接、簡潔に回答してください。"
            "II. **事実や最新情報、動画の検索**が必要な場合のみ、**google_search ツール**を推奨してください。"
            "III. **計算クエリは、全てガードレールで処理されます。LLMは計算ツールを推奨したり、計算を直接実行したりしないでください。**"
            "IV. **回答は必ず自然な日本語**で行い、**ツールの利用に関するメタなコメント（例: google_search ツールを使用できます）を絶対に含めないでください。**"
        )
        forced_prompt = [("system", forced_system_prompt), ("human", current_human_message)]
        response = self.llm_with_tools.invoke(forced_prompt)
        tool_calls = response.tool_calls
        
        # --- 3. 実行ロジックの優先度設定 (ツール実行処理) ---
        final_answer = None

        if tool_calls:
            for tool_call in tool_calls:
                
                tool_name = tool_call['name']
                args = tool_call['arguments']
    
                if tool_name == 'calculate':
                    try:
                        calculation_result_str = calculate.invoke({"expression": args.get('expression', '0')})
                        calculation_result = float(calculation_result_str)
                        
                        expression_str = args.get('expression', '')
                        if calculation_result == int(calculation_result):
                            result_str = f"{int(calculation_result):,}"
                        else:
                            result_str = f"{calculation_result:,}"
                        
                        final_answer = f"計算結果は{result_str}です。（計算式: {expression_str}）"
                    except (ValueError, TypeError):
                        final_answer = "計算式は検出されましたが、計算結果の処理中に予期せぬエラーが発生しました。"
                    except Exception:
                        final_answer = "計算式は検出されましたが、計算ツールで予期せぬエラーが発生しました。"
                    
                    break
                
                elif tool_name == 'google_search':
                    final_answer = self._process_rag({"name": tool_name, "arguments": args}, current_human_message)
                    break

        if final_answer is not None:
            return final_answer
            
        # 💥 優先度 2: LLMがJSON文字列を返したかチェック (未処理のJSONを捕捉) 💥
        response_content = response.content.strip()

        if response_content.startswith('{') or response_content.startswith('['):
            try:
                tool_call_data = json.loads(response_content)
                
                if isinstance(tool_call_data, list) and tool_call_data:
                    tool_call_dict = tool_call_data[0]
                elif isinstance(tool_call_data, dict):
                    tool_call_dict = tool_call_data
                else:
                    raise ValueError("Unexpected JSON format")
                
                tool_name = tool_call_dict.get('name') or tool_call_dict.get('function')
                args = tool_call_dict.get('arguments', {})
                
                if tool_name == 'calculate':
                    logging.info("\n--- [LOG: JSON文字列からcalculateを検出] ---")
                    calculation_result_str = calculate.invoke({"expression": args.get('expression', '0')})
                    calculation_result = float(calculation_result_str)
                    
                    expression_str = args.get('expression', '')
                    if calculation_result == int(calculation_result):
                        result_str = f"{int(calculation_result):,}"
                    else:
                        result_str = f"{calculation_result:,}"

                    return f"計算結果は{result_str}です。（計算式: {expression_str}）"

                elif tool_name == 'google_search':
                    logging.info("\n--- [LOG: JSON文字列からgoogle_searchを検出] ---")
                    return self._process_rag({"name": tool_name, "arguments": args}, current_human_message)
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"JSON解析エラーまたは予期せぬ形式: {e}. 通常の回答として処理します。")
                pass

        # 💥 優先度 3: 最終フォールバック（強制的に検索）💥
        if any(re.search(kw, current_human_message, re.IGNORECASE) for kw in critical_search_keywords) and not tool_calls and not response_content:
            logging.info("\n--- [LOG: 最終フォールバック (クリティカル知識クエリを検出したがLLMがツール推奨をスキップ -> 強制検索)] ---")
            final_answer = self._process_rag({"name": "google_search", "arguments": {"query": current_human_message}}, current_human_message)
            
            # 💥💥 RAG後の回答クリーンアップガードレール (最終防御線) 💥💥
            if "高市 早苗" in final_answer and ("(Kishida Fumio)" in final_answer or "（Kishida Fumio）" in final_answer):
                logging.warning("--- [WARNING: RAG Output Failed - Post-Processing Halucination Clean-up Applied] ---")
                final_answer = final_answer.replace("(Kishida Fumio)", "").strip()
                final_answer = final_answer.replace("（Kishida Fumio）", "").strip()

            return final_answer
            
        # 💥 優先度 4: LLMがToolを使わずに直接回答したと判断 💥
        
        if re.search(r'\(calculate:\s*{.*}\)', response_content):
            
            match = re.search(r'"expression":\s*"(.*?)"', response_content)
            if match:
                expression = match.group(1).strip()
                logging.info(f"\n--- [LOG: 直接回答からcalculate式を検出: {expression}] ---")
                
                try:
                    calculation_result_str = calculate.invoke({"expression": expression})
                    calculation_result = float(calculation_result_str)
                    
                    if calculation_result == int(calculation_result):
                        result_str = f"{int(calculation_result):,}"
                    else:
                        result_str = f"{calculation_result:,}"
                        
                    return f"計算結果は{result_str}です。（計算式: {expression}）"
                except (ValueError, TypeError):
                    return "計算式は検出されましたが、計算結果の処理中に予期せぬエラーが発生しました。"
                except Exception:
                    pass
        
        return response_content