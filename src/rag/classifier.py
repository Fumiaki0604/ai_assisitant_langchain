"""LLMベースのメッセージ分類器"""
import logging

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self, llm):
        self.llm = llm

    def _invoke(self, prompt: str) -> str:
        result = self.llm.invoke(prompt)
        return result.content if hasattr(result, 'content') else result

    def classify_message_intent(self, message: str) -> dict:
        """メッセージが質問か共有・連絡かを判定し、共有の場合は短い応答を生成"""
        try:
            prompt = """以下のメッセージの意図を判定してください。

メッセージ: {message}

判定基準:
- 質問・相談・依頼（回答や調査が必要） → question
- 情報共有・連絡・報告・お知らせ（回答不要、リアクションのみ） → share

回答は「question」か「share」の1単語のみ:"""

            result = self._invoke(prompt.format(message=message[:500]))
            intent = "share" if "share" in result.strip().lower() else "question"
            logger.info(f"Message intent: {intent}")

            if intent == "share":
                ack_prompt = """以下の社内Slackメッセージに対し、共有への感謝や前向きなリアクションを1〜2文で返してください。
丁寧すぎず、同僚に話すような自然なトーンで。絵文字は使わない。

メッセージ: {message}

応答:"""
                ack = self._invoke(ack_prompt.format(message=message[:500]))
                return {"intent": "share", "acknowledgment": ack.strip()}

            return {"intent": "question"}

        except Exception as e:
            logger.warning(f"Message intent classification failed: {e}")
            return {"intent": "question"}

    def classify_request_type(self, question: str) -> str:
        """
        Slackメッセージのリクエストタイプを判定。

        Returns:
            "knowledge"  - 知識・方法・仕様を問う質問
            "experience" - 社内実績・事例・案件経験を探す質問
            "document"   - 社内の資料・ファイルを持っている人を探す質問
            "owner"      - 特定案件・顧客の担当者を探す質問
            "opinion"    - アイデア・施策・方法論・意見を求める質問
        """
        try:
            prompt = """以下のSlackメッセージのリクエストタイプを判定してください。

メッセージ: {question}

タイプの定義:
- knowledge: 知識・仕様・使い方を問う質問（社内特有の情報）。例「GA4のイベント設定は？」「〜の方法を教えて」
- experience: 社内で過去に「担当・経験・実装した人」を探す質問（人探し）。
  例「Shopify導入事例ある？」「採用サイト担当したことある方？」「〜の事例を集めてます」
  「〜実装したことある方いますか」「〜担当された方いれば」「過去に〜やった方」
- document: 社内の資料・ファイル・テンプレートを持っている人を探す質問。例「〜資料をお持ちの方」「〜テンプレートありますか」
- owner: 特定の顧客・案件・業務の担当者を探す質問。例「PDC担当の方いますか？」「〜の担当は誰ですか？」
- opinion: アイデア・施策・方法論・意見を求める質問（一般知識でも答えられる）。
  例「効果的な施策ご存知の方いれば案ほしい」「良い方法あれば教えて」「どんなアプローチが効果的？」
  「〜についてアドバイスほしい」「これどう思う？」「何かいい案ある？」「ブレストしたい」

判定のキー：
- 「過去に〜した人」「〜経験ある方」→ experience（人探し）
- 「〜のアイデア・施策・方法・案がほしい」→ opinion（一般知識で答えられるアドバイス）
- 「特定案件/変数/仕様がわからない」→ knowledge または experience
- 単純な仕様・使い方は knowledge

回答は「knowledge」「experience」「document」「owner」「opinion」のいずれか1単語のみ:"""

            result = self._invoke(prompt.format(question=question[:500]))
            result_lower = result.strip().lower()
            for t in ("experience", "document", "owner", "opinion"):
                if t in result_lower:
                    return t
            return "knowledge"
        except Exception as e:
            logger.warning(f"Slack request type classification failed: {e}")
            return "knowledge"

    def classify_question_type(self, question: str) -> str:
        """質問が内部製品に関するものか外部サービスに関するものかを判定"""
        try:
            prompt = """以下の質問が「自社製品(ecbeing/メルカート/visumo等)に関する技術的な質問」か「外部サービス/一般的な技術の質問」かを判定してください。

質問: {question}

判定基準:
- 自社製品の機能、設定、運用、事例に関する質問 → internal
- Google、Bing、AWS等の外部サービスに関する質問 → external
- SEO、インデックス、サーバー等の一般技術で自社製品に限定されない質問 → external

回答は「internal」か「external」の1単語のみ:"""

            result = self._invoke(prompt.format(question=question))
            classification = "external" if "external" in result.lower() else "internal"
            logger.info(f"Question classified as: {classification}")
            return classification
        except Exception as e:
            logger.warning(f"Question classification failed: {e}")
            return "internal"
