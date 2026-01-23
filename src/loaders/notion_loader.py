"""
NotionページをPineconeに登録するローダー
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import NotionDBLoader
from src.rag.embeddings import get_embeddings
from config.settings import settings
import logging
import requests

logger = logging.getLogger(__name__)


class NotionLoader:
    """
    NotionページをPineconeに登録するクラス
    """

    def __init__(self, api_token: str = None):
        self.api_token = api_token or getattr(settings, 'notion_api_key', None)
        if not self.api_token:
            raise ValueError("Notion API token is required. Set NOTION_API_TOKEN in .env")

        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

    def search_pages(self, query: str = None) -> list:
        """
        Notionページを検索
        """
        url = "https://api.notion.com/v1/search"
        payload = {
            "filter": {"property": "object", "value": "page"},
            "page_size": 100
        }
        if query:
            payload["query"] = query

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            pages = data.get("results", [])
            print(f"[DEBUG] Found {len(pages)} pages in Notion")

            # デバッグ: 取得したページのタイトルを表示
            for page in pages:
                title = self.get_page_title(page)
                print(f"  - Page found: {title} (ID: {page.get('id')})")

            return pages
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search Notion pages: {e}")
            return []

    def get_page_content(self, page_id: str, debug: bool = False) -> str:
        """
        ページのブロックコンテンツを取得
        """
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        content_parts = []

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            blocks = data.get("results", [])
            if debug:
                print(f"[DEBUG] Found {len(blocks)} blocks in page")
                for b in blocks[:10]:
                    btype = b.get('type')
                    print(f"  - Block type: {btype}")
                    if btype == 'paragraph':
                        print(f"    Data: {b.get('paragraph', {})}")

            for block in blocks:
                block_type = block.get("type")
                block_data = block.get(block_type, {})

                # テキストを抽出
                text = self._extract_text_from_block(block_type, block_data)
                if text:
                    content_parts.append(text)

            return "\n\n".join(content_parts)

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get page content: {e}")
            return ""

    def _extract_text_from_block(self, block_type: str, block_data: dict) -> str:
        """
        ブロックからテキストを抽出
        """
        text_types = [
            "paragraph", "heading_1", "heading_2", "heading_3",
            "bulleted_list_item", "numbered_list_item", "quote",
            "callout", "toggle"
        ]

        if block_type in text_types:
            rich_text = block_data.get("rich_text", [])
            return "".join([t.get("plain_text", "") for t in rich_text])

        elif block_type == "code":
            rich_text = block_data.get("rich_text", [])
            code = "".join([t.get("plain_text", "") for t in rich_text])
            language = block_data.get("language", "")
            return f"```{language}\n{code}\n```"

        elif block_type == "to_do":
            rich_text = block_data.get("rich_text", [])
            checked = block_data.get("checked", False)
            text = "".join([t.get("plain_text", "") for t in rich_text])
            return f"[{'x' if checked else ' '}] {text}"

        return ""

    def get_page_title(self, page: dict) -> str:
        """
        ページタイトルを取得
        """
        properties = page.get("properties", {})

        # タイトルプロパティを探す
        for prop_name, prop_value in properties.items():
            if prop_value.get("type") == "title":
                title_array = prop_value.get("title", [])
                if title_array:
                    return "".join([t.get("plain_text", "") for t in title_array])

        return "Untitled"

    def load_pages(self, query: str = None, limit: int = 50) -> list:
        """
        Notionページを読み込んでドキュメント形式に変換
        """
        pages = self.search_pages(query)[:limit]
        documents = []

        for page in pages:
            page_id = page.get("id")
            title = self.get_page_title(page)
            url = page.get("url", "")

            # ページコンテンツを取得
            content = self.get_page_content(page_id)
            print(f"[DEBUG] Page '{title}': content length = {len(content)}")

            if content:
                documents.append({
                    "content": f"# {title}\n\n{content}",
                    "metadata": {
                        "source": "notion",
                        "page_id": page_id,
                        "title": f"Notion: {title}",
                        "url": url
                    }
                })
                logger.info(f"Loaded page: {title}")

        logger.info(f"Total: Loaded {len(documents)} pages from Notion")
        return documents

    def save_to_pinecone(self, documents: list) -> bool:
        """
        ドキュメントをPineconeに保存
        """
        if not documents:
            logger.warning("No documents to save")
            return False

        try:
            embeddings = get_embeddings()

            all_texts = []
            all_metadatas = []

            for doc in documents:
                chunks = self.text_splitter.split_text(doc["content"])
                for i, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    metadata = doc["metadata"].copy()
                    metadata["chunk_id"] = i
                    all_metadatas.append(metadata)

            logger.info(f"Saving {len(all_texts)} chunks to Pinecone...")

            PineconeVectorStore.from_texts(
                texts=all_texts,
                embedding=embeddings,
                metadatas=all_metadatas,
                index_name=settings.pinecone_index_name
            )

            logger.info("Documents saved to Pinecone successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save to Pinecone: {e}", exc_info=True)
            return False


def load_notion_pages(query: str = None, limit: int = 50):
    """
    Notionページを読み込んでPineconeに保存
    """
    try:
        loader = NotionLoader()
        documents = loader.load_pages(query, limit)

        if documents:
            success = loader.save_to_pinecone(documents)
            return success, len(documents)
        return False, 0

    except ValueError as e:
        logger.error(f"Notion loader error: {e}")
        return False, 0


def load_notion_page_by_id(page_id: str):
    """
    特定のページIDを指定して読み込み
    """
    try:
        loader = NotionLoader()

        # ページ情報を取得
        url = f"https://api.notion.com/v1/pages/{page_id}"
        response = requests.get(url, headers=loader.headers)
        response.raise_for_status()
        page = response.json()

        title = loader.get_page_title(page)
        page_url = page.get("url", "")
        content = loader.get_page_content(page_id, debug=True)

        if not content:
            print(f"ページ '{title}' にはコンテンツがありません")
            return False, 0

        documents = [{
            "content": f"# {title}\n\n{content}",
            "metadata": {
                "source": "notion",
                "page_id": page_id,
                "title": f"Notion: {title}",
                "url": page_url
            }
        }]

        print(f"ページ '{title}' を読み込みました (コンテンツ長: {len(content)})")

        success = loader.save_to_pinecone(documents)
        return success, 1 if success else 0

    except Exception as e:
        print(f"エラー: {e}")
        return False, 0


if __name__ == "__main__":
    import logging
    import sys
    logging.basicConfig(level=logging.INFO)

    # コマンドライン引数でページIDまたは検索クエリが指定された場合
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # --search オプションで検索
        if arg == "--search" and len(sys.argv) > 2:
            query = sys.argv[2]
            print(f"\n'{query}' を検索中...")
            loader = NotionLoader()
            pages = loader.search_pages(query)
            print(f"\n検索結果: {len(pages)} 件")
            for page in pages[:20]:
                title = loader.get_page_title(page)
                page_id = page.get("id")
                print(f"  - {title}")
                print(f"    ID: {page_id}")
            sys.exit(0)

        # ページIDが指定された場合
        page_id = arg
        print(f"\nページID {page_id} を読み込み中...")
        success, count = load_notion_page_by_id(page_id)
    else:
        print("\nNotionページを検索中...")
        success, count = load_notion_pages(limit=50)

    if success:
        print(f"\n完了: {count} 件のページをPineconeに登録しました")
    else:
        print("\nNotionからのデータ取得に失敗しました")
        print("NOTION_API_TOKEN を .env に設定してください")
