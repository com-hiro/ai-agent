"""
Ollama Tool Agentのメイン実行エントリーポイント。

ログ設定の初期化、AdaptiveAgentの起動、およびユーザー入力を処理する
メインループを提供する。
"""
import os
import logging
import sys
from agent_core import AdaptiveAgent

# --------------------------------------------------------------------------
# --- ログ設定と環境変数 ---
# --------------------------------------------------------------------------

LOG_FILE_NAME = "agent_session_history.log"
# 環境変数 'AGENT_LOG_ENABLED' が '1', 'true', 'True', 'TRUE' のいずれかに設定されていればログON
LOG_ENABLED = os.environ.get("AGENT_LOG_ENABLED", "0") in ["1", "true", "True", "TRUE"]

if LOG_ENABLED:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE_NAME, mode='a', encoding='utf-8')
        ]
    )
    logging.getLogger().info("✅ Agent Log Output: ON")
else:
    logging.basicConfig(level=logging.CRITICAL, handlers=[logging.NullHandler()])


def main():
    """
    エージェントを初期化し、ユーザーとの対話ループを開始する。

    Ctrl/Cmd+Cが入力されるまで、ユーザーからの入力を受け付け、
    AdaptiveAgentの処理結果を出力する。
    """
    agent = AdaptiveAgent(model_name="mistral:instruct") 
    
    if not LOG_ENABLED:
        print(f"🔔 Agent Log Output: OFF")
    
    is_search_active = os.environ.get("SERPAPI_API_KEY") is not None
    if not is_search_active:
        if LOG_ENABLED: logging.warning("⚠️ SerpAPIキーが未設定です。検索機能は動作しません。")
        else: print("⚠️ SerpAPIキーが未設定です。検索機能は動作しません。")


    if LOG_ENABLED:
        logging.info("Ollama Tool Agent 起動（安全・モジュール化版、終了するには Ctrl/Cmd+C）")
    else:
        print("\nOllama Tool Agent 起動（終了するには Ctrl+C）")
        print("例: 100を5で割って、それに20を足すといくつ？")

    
    while True:
        try:
            user_input = input("あなた: ")
            
            if LOG_ENABLED: logging.info(f"\n>>> USER INPUT: {user_input} <<<")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            
            answer = agent.run(user_input) 
            
            if LOG_ENABLED: logging.info(f"\n--- Agent Final Answer: {answer} ---")
            
            print("\nAgent:", answer)
            print("---")
            
        except KeyboardInterrupt:
            print("\nエージェントを終了します。")
            break
        except Exception as e:
            if LOG_ENABLED: logging.critical(f"クリティカルなエラーが発生しました: {e}", exc_info=True)
            print(f"クリティカルなエラーが発生しました: {e}")
            break

if __name__ == "__main__":
    main()