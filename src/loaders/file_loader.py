"""
ファイル（PDF/Word/Markdown）をPineconeに登録するローダー
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from src.loaders.pinecone_storage import save_to_pinecone as _save_to_pinecone
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class FileDocumentLoader:
    """
    ファイルをPineconeに登録するクラス
    """

    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.docx': 'word',
        '.doc': 'word',
        '.md': 'markdown',
        '.txt': 'text'
    }

    def __init__(self, documents_dir: str = None):
        self.documents_dir = documents_dir or os.path.join(
            os.path.dirname(__file__), '../../documents'
        )

    def get_loader_for_file(self, file_path: str):
        """
        ファイル拡張子に応じたローダーを返す
        """
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext in ['.docx', '.doc']:
            return Docx2txtLoader(file_path)
        elif ext == '.md':
            return TextLoader(file_path, encoding='utf-8')
        elif ext == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        else:
            return None

    def _extract_frontmatter(self, file_path: str) -> dict:
        """YAMLフロントマターをメタデータとして抽出"""
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if not match:
                return {}
            metadata = {}
            for line in match.group(1).splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip()
            return metadata
        except Exception:
            return {}

    def load_file(self, file_path: str) -> list:
        """
        単一ファイルを読み込み
        """
        try:
            loader = self.get_loader_for_file(file_path)
            if loader is None:
                logger.warning(f"Unsupported file type: {file_path}")
                return []

            documents = loader.load()
            file_name = Path(file_path).name
            frontmatter = self._extract_frontmatter(file_path)

            # メタデータを追加
            for doc in documents:
                doc.metadata.update({
                    "source": frontmatter.get("source", "file"),
                    "file_name": file_name,
                    "file_path": file_path,
                    "title": file_name,
                    **{k: v for k, v in frontmatter.items() if k != "source"}
                })

            logger.info(f"Loaded {len(documents)} pages from {file_name}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return []

    def load_directory(self, directory: str = None) -> list:
        """
        ディレクトリ内の全ファイルを読み込み
        """
        target_dir = directory or self.documents_dir

        if not os.path.exists(target_dir):
            logger.error(f"Directory not found: {target_dir}")
            return []

        all_documents = []

        for root, dirs, files in os.walk(target_dir):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    documents = self.load_file(file_path)
                    all_documents.extend(documents)

        logger.info(f"Total: Loaded {len(all_documents)} documents from {target_dir}")
        return all_documents

    def save_to_pinecone(self, documents: list) -> bool:
        """
        ドキュメントをPineconeに保存
        """
        return _save_to_pinecone(documents)


def load_documents_from_directory(directory: str = None):
    """
    ディレクトリからドキュメントを読み込んでPineconeに保存
    """
    loader = FileDocumentLoader(directory)
    documents = loader.load_directory()

    if documents:
        success = loader.save_to_pinecone(documents)
        return success, len(documents)
    return False, 0


def load_single_file(file_path: str):
    """
    単一ファイルを読み込んでPineconeに保存
    """
    loader = FileDocumentLoader()
    documents = loader.load_file(file_path)

    if documents:
        success = loader.save_to_pinecone(documents)
        return success, len(documents)
    return False, 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    documents_dir = os.path.join(os.path.dirname(__file__), '../../documents')
    print(f"\nドキュメントディレクトリ: {documents_dir}")

    # ディレクトリ内のファイルを確認
    if os.path.exists(documents_dir):
        files = os.listdir(documents_dir)
        if files:
            print(f"見つかったファイル: {files}")
            success, count = load_documents_from_directory(documents_dir)
            if success:
                print(f"\n完了: {count} 件のドキュメントをPineconeに登録しました")
        else:
            print("ディレクトリにファイルがありません")
            print("documents/ フォルダにPDF、Word、Markdownファイルを配置してください")
    else:
        print(f"ディレクトリが存在しません: {documents_dir}")
