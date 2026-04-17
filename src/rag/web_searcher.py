"""Web検索・URLコンテンツ取得"""
import re
import requests
import logging

logger = logging.getLogger(__name__)


def fetch_url_content(url: str) -> str:
    """URLの内容を取得してテキストを返す"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        text = re.sub(r'<[^>]+>', '', response.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:2000]
    except Exception as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        return ""


def web_search(query: str, num_results: int = 3, fetch_content: bool = True) -> list:
    """DuckDuckGoでWeb検索を実行し、上位結果のコンテンツも取得"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {'q': query, 'kl': 'jp-jp'}
        response = requests.get(
            'https://html.duckduckgo.com/html/',
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()

        results = []
        pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, response.text)

        for url, title in matches[:num_results]:
            if url.startswith('//duckduckgo.com/l/?uddg='):
                url = requests.utils.unquote(url.split('uddg=')[1].split('&')[0])
            results.append({'url': url, 'title': title.strip()})

        if fetch_content:
            for result in results[:2]:
                result['content'] = fetch_url_content(result['url'])

        logger.info(f"Web search found {len(results)} results for: {query[:30]}...")
        return results
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []
