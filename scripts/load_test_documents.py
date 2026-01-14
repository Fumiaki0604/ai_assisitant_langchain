"""
テストドキュメントをPineconeに登録
"""
import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from src.rag.embeddings import get_embeddings
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# サンプルドキュメント
SAMPLE_DOCUMENTS = [
    {
        "content": """
# 会社の勤務時間ポリシー

## 標準勤務時間
- 勤務時間: 9:00 - 18:00（休憩1時間を含む）
- コアタイム: 10:00 - 15:00
- フレックスタイム制度を採用しています

## リモートワーク
- 週3日までリモートワーク可能
- 事前にSlackで上長に報告すること
- リモート時もコアタイムは必ず稼働すること

## 休暇制度
- 年次有給休暇: 入社時10日付与（最大20日）
- 夏季休暇: 3日
- 年末年始休暇: 12/29 - 1/3
        """,
        "metadata": {"source": "company_policy", "title": "勤務時間ポリシー"}
    },
    {
        "content": """
# 経費精算ガイドライン

## 交通費
- 公共交通機関の利用を原則とする
- タクシー利用は22時以降のみ承認
- 領収書の添付必須

## 出張費
- 宿泊費: 1泊10,000円まで
- 食事代: 1日3,000円まで
- 出張申請は5営業日前までに提出

## 申請方法
1. 経費精算システムにログイン
2. 該当する経費項目を選択
3. 領収書をアップロード
4. 上長の承認を得る
        """,
        "metadata": {"source": "company_policy", "title": "経費精算ガイドライン"}
    },
    {
        "content": """
# ITシステム利用規程

## パスワードポリシー
- 最低12文字以上
- 英大文字、小文字、数字、記号を含むこと
- 90日ごとに変更必須

## セキュリティ
- 会社支給PCは暗号化必須
- USBメモリの使用は禁止
- 機密情報をクラウドストレージに保存しない

## ソフトウェア
- 承認されたソフトウェアのみインストール可能
- 個人用ソフトウェアのインストール禁止
- ライセンス違反に注意
        """,
        "metadata": {"source": "company_policy", "title": "ITシステム利用規程"}
    },
    {
        "content": """
# よくある質問（FAQ）

## Q: 有給休暇の申請方法は？
A: 勤怠管理システムから申請し、上長の承認を得てください。最低3日前までに申請することを推奨します。

## Q: リモートワークの申請は？
A: 当日朝9時までにSlackの #remote-work チャンネルで報告してください。事前申請は不要です。

## Q: 経費精算の締め日は？
A: 毎月月末が締め日です。翌月10日までに申請してください。

## Q: 社員証を紛失した場合は？
A: すぐに総務部に連絡し、再発行手続きを行ってください。再発行には2,000円かかります。

## Q: プロジェクトの予算確認方法は？
A: プロジェクト管理ツールの「予算」タブから確認できます。詳細は経理部にお問い合わせください。
        """,
        "metadata": {"source": "faq", "title": "よくある質問"}
    },
    {
        "content": """
# AWS Bedrock使用ガイド

## 概要
AWS BedrockはAmazonが提供するフルマネージドなLLMサービスです。

## 利用可能なモデル
- Claude 3.5 Sonnet: 最も高性能なモデル
- Claude 3 Haiku: 高速・低コストなモデル
- Titan Embeddings: テキスト埋め込み生成

## 料金
- Claude 3.5 Sonnet: $0.003/1K入力トークン、$0.015/1K出力トークン
- 埋め込み生成は従量課金

## 使用方法
1. AWSコンソールでBedrockサービスにアクセス
2. モデルアクセスを有効化
3. boto3またはLangChainから呼び出し
        """,
        "metadata": {"source": "technical_doc", "title": "AWS Bedrock使用ガイド"}
    }
]


def load_documents_to_pinecone():
    """
    サンプルドキュメントをPineconeに登録
    """
    try:
        logger.info("テストドキュメントの読み込み開始...")

        # テキストスプリッターを初期化
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )

        # すべてのドキュメントを処理
        all_texts = []
        all_metadatas = []

        for doc in SAMPLE_DOCUMENTS:
            # ドキュメントをチャンクに分割
            chunks = text_splitter.split_text(doc["content"])
            logger.info(f"ドキュメント '{doc['metadata']['title']}' を {len(chunks)} チャンクに分割")

            # 各チャンクにメタデータを付与
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                metadata = doc["metadata"].copy()
                metadata["chunk_id"] = i
                all_metadatas.append(metadata)

        logger.info(f"合計 {len(all_texts)} チャンクを準備完了")

        # 埋め込みモデルを取得
        embeddings = get_embeddings()

        # Pineconeに保存
        logger.info("Pineconeへの保存開始...")

        # 環境変数を設定（PineconeVectorStoreが内部で参照する）
        os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

        vectorstore = PineconeVectorStore.from_texts(
            texts=all_texts,
            embedding=embeddings,
            metadatas=all_metadatas,
            index_name=settings.pinecone_index_name
        )

        logger.info("✅ ドキュメントの登録完了")

        # インデックス統計を確認
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        stats = index.describe_index_stats()

        print("\n" + "="*50)
        print("✅ テストドキュメント登録完了！")
        print(f"登録されたベクトル数: {stats.get('total_vector_count', 0)}")
        print(f"ドキュメント数: {len(SAMPLE_DOCUMENTS)}")
        print(f"チャンク数: {len(all_texts)}")
        print("="*50 + "\n")

        return True

    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        print(f"\n❌ エラー: {e}\n")
        return False


if __name__ == "__main__":
    load_documents_to_pinecone()
